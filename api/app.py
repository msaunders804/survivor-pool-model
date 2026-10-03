"""Phase 9 API. Three endpoints, matching the wireframe's v1 scope.

Deliberately thin: nothing here runs a simulation. The weekly job
(run_weekly.py, hers, unmodified) writes pick_sheets/latest.csv;
ingest.py loads that into `recommendations`; these endpoints just read
the database. Heavy compute stays out of the request path by construction.
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from auth import check_password, issue_token, require_auth
from models import Entry, League, Ownership, Pick, Recommendation, SessionLocal, init_db

try:
    from survivor.data.team_keys import to_abbreviation
except ImportError:
    to_abbreviation = None  # her package not installed in this environment -- fall back to raw text

# Her to_abbreviation only knows full names ("Kansas City Chiefs") and
# existing abbreviations -- not bare nicknames. Splash's own export uses
# nicknames only ("Chiefs", "Seahawks"), so without this map almost every
# real paste would silently fail to resolve and never match a team on the
# board. Kept here, not in her package -- this is an ownership-paste
# concern, not a general team-name one.
NICKNAME_TO_ABBREVIATION = {
    "CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF",
    "PANTHERS": "CAR", "BEARS": "CHI", "BENGALS": "CIN", "BROWNS": "CLE",
    "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
    "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAX", "JAGS": "JAX",
    "CHIEFS": "KC", "RAIDERS": "LV", "CHARGERS": "LAC", "RAMS": "LAR",
    "DOLPHINS": "MIA", "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO",
    "GIANTS": "NYG", "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT",
    "49ERS": "SF", "NINERS": "SF", "SEAHAWKS": "SEA", "BUCCANEERS": "TB",
    "BUCS": "TB", "TITANS": "TEN", "COMMANDERS": "WAS",
}

app = FastAPI(title="Survivor Pool API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PROTECTED = [Depends(require_auth)]  # pass as dependencies=PROTECTED on every /leagues/* route


@app.on_event("startup")
def startup() -> None:
    init_db()


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
        pick = db.query(Pick).filter_by(entry_id_fk=entry.id, week=body.week).one_or_none()
        if pick is None:
            pick = Pick(entry_id_fk=entry.id, week=body.week)
            db.add(pick)
        pick.team = body.team
        pick.confirmed = body.confirmed
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
        return LeagueConfigOut(
            name=league.name, entry_cap=league.entry_cap, buy_in=league.buy_in,
            rake_pct=league.rake_pct, pot=league.pot, week_start=league.week_start,
            week_end=league.week_end, n_rivals=league.n_rivals, current_week=league.current_week,
        )
    finally:
        db.close()


# ---------- 4. league ownership (who's picked what, reported from Splash) ----------

LINE_RE = re.compile(r"^(?P<team>.+?)[\s,]+(?P<a>[\d,]+)\s*%?\s*(?P<b>[\d,]+)?\s*%?\s*$")


def parse_ownership_text(text: str) -> list[dict]:
    """Handles 'TEAM COUNT', 'TEAM PCT%', and 'TEAM PCT% COUNT' (Splash's
    own Stats-tab order) per line. When both a percentage and a count are
    present, the count wins -- it's the more precise of the two. Lines
    that don't parse are skipped, not errored on, so one bad paste doesn't
    block the rest."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        team_raw = m.group("team").strip()
        a = int(m.group("a").replace(",", ""))
        b = m.group("b")
        if b:
            # "TEAM PCT% COUNT" -- a is the percentage, b is the count
            pct, count = float(a), int(b.replace(",", ""))
        elif "%" in line:
            pct, count = float(a), None
        else:
            pct, count = None, a
        team = team_raw.upper()
        if to_abbreviation:
            try:
                team = to_abbreviation(team_raw)
            except KeyError:
                team = NICKNAME_TO_ABBREVIATION.get(team.replace(" ", ""), team)
        rows.append({"team": team, "pct": pct, "pick_count": count})
    return rows


class OwnershipIn(BaseModel):
    week: int
    raw_text: str


class OwnershipRow(BaseModel):
    team: str
    pick_count: int | None
    pct: float | None


@app.post("/leagues/{league_id}/ownership", response_model=list[OwnershipRow], dependencies=PROTECTED)
def save_ownership(league_id: int, body: OwnershipIn):
    rows = parse_ownership_text(body.raw_text)
    if not rows:
        raise HTTPException(400, "couldn't parse any lines -- expected 'TEAM COUNT' or 'TEAM PCT% COUNT' per line")
    db = SessionLocal()
    try:
        for row in rows:
            existing = (
                db.query(Ownership)
                .filter_by(league_id=league_id, week=body.week, team=row["team"])
                .one_or_none()
            )
            if existing is None:
                existing = Ownership(league_id=league_id, week=body.week, team=row["team"])
                db.add(existing)
            existing.pick_count = row["pick_count"]
            existing.pct = row["pct"]
        db.commit()
        return [OwnershipRow(**r) for r in rows]
    finally:
        db.close()


@app.get("/leagues/{league_id}/ownership", response_model=list[OwnershipRow], dependencies=PROTECTED)
def get_ownership(league_id: int, week: int):
    db = SessionLocal()
    try:
        rows = db.query(Ownership).filter_by(league_id=league_id, week=week).all()
        return [OwnershipRow(team=r.team, pick_count=r.pick_count, pct=r.pct) for r in rows]
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
