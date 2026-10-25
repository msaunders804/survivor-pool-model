"""Phase 9 API. Three endpoints, matching the wireframe's v1 scope.

Deliberately thin: nothing here runs a simulation. The weekly job
(run_weekly.py, hers, unmodified) writes pick_sheets/latest.csv;
ingest.py loads that into `recommendations`; these endpoints just read
the database. Heavy compute stays out of the request path by construction.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ownership_parse import parse_ownership_text
from survivor.data.availability import parse_availability
from pick_rules import validate_pick
from seed import seed_if_empty
from survivor.data.team_keys import ALL_TEAMS
from auth import check_password, issue_token, require_auth
from models import Entry, League, Ownership, Pick, Recommendation, SessionLocal, init_db
from nfl import advance_league_week, week_games
from auto_results import fill_pending_results

app = FastAPI(title="Survivor Pool API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PROTECTED = [Depends(require_auth)]  # pass as dependencies=PROTECTED on every /leagues/* route


@app.on_event("startup")
def startup() -> None:
    init_db()
    seed_if_empty()  # first boot on a fresh database: create the league and entries


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class LoginIn(BaseModel):
    password: str


class LoginOut(BaseModel):
    token: str


@app.post("/login", response_model=LoginOut)
def login(body: LoginIn):
    if not check_password(body.password):
        raise HTTPException(401, "wrong password")
    return LoginOut(token=issue_token())


# ---------- 1. current recommendation ----------

class EntryPick(BaseModel):
    entry_id: str
    team: str
    used_teams_before: list[str]


class RecommendationOut(BaseModel):
    week: int
    expected_total_payout: float | None
    payout_se: float | None
    objective: str | None = None
    win_probability: float | None = None
    picks: list[EntryPick]


@app.get("/leagues/{league_id}/recommendation", response_model=RecommendationOut, dependencies=PROTECTED)
def get_recommendation(league_id: int, week: int):
    db = SessionLocal()
    try:
        rows = db.query(Recommendation).filter_by(league_id=league_id, week=week).all()
        if not rows:
            raise HTTPException(404, f"no recommendation stored for league {league_id}, week {week}")
        return RecommendationOut(
            week=week,
            expected_total_payout=rows[0].expected_total_payout,
            payout_se=rows[0].payout_se,
            objective=rows[0].objective,
            win_probability=rows[0].win_probability,
            picks=[
                EntryPick(
                    entry_id=r.entry_id, team=r.team,
                    used_teams_before=[t for t in r.used_teams_before.split(",") if t],
                )
                for r in sorted(rows, key=lambda r: r.entry_id)
            ],
        )
    finally:
        db.close()


# ---------- 2. entries + pick history ----------

class PickOut(BaseModel):
    week: int
    team: str
    survived: bool | None
    confirmed: bool


class EntryOut(BaseModel):
    entry_id: str
    picks: list[PickOut]


@app.get("/leagues/{league_id}/entries", response_model=list[EntryOut], dependencies=PROTECTED)
def get_entries(league_id: int):
    db = SessionLocal()
    try:
        entries = db.query(Entry).filter_by(league_id=league_id).order_by(Entry.entry_id).all()
        return [
            EntryOut(
                entry_id=e.entry_id,
                picks=[
                    PickOut(week=p.week, team=p.team, survived=p.survived, confirmed=p.confirmed)
                    for p in sorted(e.picks, key=lambda p: p.week)
                ],
            )
            for e in entries
        ]
    finally:
        db.close()


@app.get("/teams", response_model=list[str], dependencies=PROTECTED)
def get_teams():
    """Every NFL team abbreviation, for the pick dropdowns."""
    return ALL_TEAMS


class SavePickIn(BaseModel):
    entry_id: str
    week: int
    team: str
    confirmed: bool = False


@app.post("/leagues/{league_id}/picks", dependencies=PROTECTED)
def save_pick(league_id: int, body: SavePickIn):
    """The one write endpoint in v1. Writes to `picks`, never to
    `recommendations` -- the model's output and the entrant's actual
    choice stay separate tables, which is what makes 'reset to the
    model' possible later."""
    db = SessionLocal()
    try:
        entry = db.query(Entry).filter_by(league_id=league_id, entry_id=body.entry_id).one_or_none()
        if entry is None:
            raise HTTPException(404, f"entry {body.entry_id!r} not found in league {league_id}")
        used_before = [p.team for p in entry.picks if p.week < body.week]
        try:
            team = validate_pick(body.team, used_before)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        pick = db.query(Pick).filter_by(entry_id_fk=entry.id, week=body.week).one_or_none()
        if pick is None:
            pick = Pick(entry_id_fk=entry.id, week=body.week)
            db.add(pick)
        pick.team = team
        pick.confirmed = body.confirmed
        db.commit()
        return {"ok": True}
    finally:
        db.close()


class ResultIn(BaseModel):
    entry_id: str
    week: int
    survived: bool | None = None  # None = back to pending


@app.post("/leagues/{league_id}/results", dependencies=PROTECTED)
def save_result(league_id: int, body: ResultIn):
    """Mark whether an entry's pick won (survived) or lost that week. The
    weekly job reads this back: an entry with a loss is dropped from planning."""
    db = SessionLocal()
    try:
        entry = db.query(Entry).filter_by(league_id=league_id, entry_id=body.entry_id).one_or_none()
        if entry is None:
            raise HTTPException(404, f"entry {body.entry_id!r} not found in league {league_id}")
        pick = db.query(Pick).filter_by(entry_id_fk=entry.id, week=body.week).one_or_none()
        if pick is None:
            raise HTTPException(404, f"no pick saved for {body.entry_id!r} in week {body.week}")
        pick.survived = body.survived
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ---------- 3. league config ----------

class LeagueConfigOut(BaseModel):
    name: str
    entry_cap: int
    buy_in: float
    rake_pct: float
    pot: float
    week_start: int
    week_end: int
    n_rivals: int
    current_week: int


class LeagueConfigIn(BaseModel):
    entry_cap: int | None = None
    buy_in: float | None = None
    rake_pct: float | None = None
    pot: float | None = None
    week_start: int | None = None
    week_end: int | None = None
    n_rivals: int | None = None
    current_week: int | None = None


@app.get("/leagues/{league_id}/config", response_model=LeagueConfigOut, dependencies=PROTECTED)
def get_config(league_id: int):
    db = SessionLocal()
    try:
        league = db.get(League, league_id)
        if league is None:
            raise HTTPException(404, f"league {league_id} not found")
        advance_league_week(db, league)  # the week turns by itself each Tuesday; no one has to bump it
        fill_pending_results(db, league)  # Won/Lost from final scores, for picks still pending
        return LeagueConfigOut(
            name=league.name, entry_cap=league.entry_cap, buy_in=league.buy_in,
            rake_pct=league.rake_pct, pot=league.pot, week_start=league.week_start,
            week_end=league.week_end, n_rivals=league.n_rivals, current_week=league.current_week,
        )
    finally:
        db.close()


@app.get("/leagues/{league_id}/schedule", dependencies=PROTECTED)
def get_schedule(league_id: int, week: int):
    """That week's games and byes, for the matchup labels. Empty (not an error) if ESPN
    can't be reached, so the pages still load without matchups."""
    db = SessionLocal()
    try:
        league = db.get(League, league_id)
        if league is None:
            raise HTTPException(404, f"league {league_id} not found")
        year = league.year
    finally:
        db.close()
    try:
        return week_games(week, year)
    except Exception:
        return {"week": week, "games": [], "byes": [], "unavailable": True}


# ---------- 4. league ownership (who's picked what, reported from Splash) ----------

class OwnershipIn(BaseModel):
    week: int
    raw_text: str


class OwnershipRow(BaseModel):
    team: str
    pick_count: int | None = None
    pct: float | None = None
    available: int | None = None  # availability report rows: live entries that can still pick the team
    n_alive: int | None = None


def _parse_pasted_table(raw_text: str) -> list[dict]:
    """Brent's paste is Splash's availability report ('Vikings: 387/610 (63%)'); the older
    pick-share formats ('Chiefs 39% 948') still work."""
    try:
        n_alive, available = parse_availability(raw_text, allow_missing=True)
        return [{"team": t, "available": c, "n_alive": n_alive} for t, c in available.items()]
    except ValueError:
        return parse_ownership_text(raw_text)


@app.post("/leagues/{league_id}/ownership", response_model=list[OwnershipRow], dependencies=PROTECTED)
def save_ownership(league_id: int, body: OwnershipIn):
    rows = _parse_pasted_table(body.raw_text)
    if not rows:
        raise HTTPException(400, "couldn't parse any lines -- expected 'TEAM: AVAILABLE/LIVE (PCT%)', 'TEAM COUNT' or 'TEAM PCT% COUNT' per line")
    db = SessionLocal()
    try:
        league = db.get(League, league_id)
        if league is None:
            raise HTTPException(404, f"league {league_id} not found")
        # Re-derive the current week right now rather than trusting body.week: the page fetches
        # its week once on load, so a tab left open across the Tuesday turnover would otherwise
        # tag the paste to the week that just ended, and the weekly job would never find it.
        advance_league_week(db, league)
        week = league.current_week
        # a re-paste replaces the whole week's table, so a team dropped from it doesn't linger
        db.query(Ownership).filter_by(league_id=league_id, week=week).delete()
        for row in rows:
            existing = (
                db.query(Ownership)
                .filter_by(league_id=league_id, week=week, team=row["team"])
                .one_or_none()
            )
            if existing is None:
                existing = Ownership(league_id=league_id, week=week, team=row["team"])
                db.add(existing)
            existing.pick_count = row.get("pick_count")
            existing.pct = row.get("pct")
            existing.available = row.get("available")
            existing.n_alive = row.get("n_alive")
        if rows[0].get("n_alive"):
            league.n_rivals = rows[0]["n_alive"]  # keep the fallback rival count in step with the latest report
        db.commit()
        return [OwnershipRow(**r) for r in rows]
    finally:
        db.close()


@app.get("/leagues/{league_id}/ownership", response_model=list[OwnershipRow], dependencies=PROTECTED)
def get_ownership(league_id: int, week: int):
    db = SessionLocal()
    try:
        rows = db.query(Ownership).filter_by(league_id=league_id, week=week).all()
        return [OwnershipRow(team=r.team, pick_count=r.pick_count, pct=r.pct, available=r.available, n_alive=r.n_alive)
                for r in rows]
    finally:
        db.close()


# ---------- 5. league config ----------

@app.patch("/leagues/{league_id}/config", dependencies=PROTECTED)
def update_config(league_id: int, body: LeagueConfigIn):
    db = SessionLocal()
    try:
        league = db.get(League, league_id)
        if league is None:
            raise HTTPException(404, f"league {league_id} not found")
        for field, value in body.model_dump(exclude_none=True).items():
            setattr(league, field, value)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# Serve the web/ pages from the same service, so one Render web service is the
# whole app. Mounted last so it never shadows an API route.
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")
