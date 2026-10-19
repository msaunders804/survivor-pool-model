import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")
API = str(Path(__file__).resolve().parent.parent / "api")


def game(home, away, home_score, away_score, completed=True):
    return {"competitions": [{
        "status": {"type": {"completed": completed}},
        "competitors": [
            {"homeAway": "home", "score": str(home_score), "team": {"displayName": home}},
            {"homeAway": "away", "score": str(away_score), "team": {"displayName": away}},
        ],
    }]}


@pytest.fixture
def mods(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    monkeypatch.syspath_prepend(API)
    for name in ("models", "nfl", "auto_results"):
        sys.modules.pop(name, None)
    models = importlib.import_module("models")
    models.init_db()
    return models, importlib.import_module("nfl"), importlib.import_module("auto_results")


@pytest.fixture
def league_with_picks(mods):
    models, _, _ = mods
    db = models.SessionLocal()
    user = models.User(email="a@example.com", password_hash="x")
    db.add(user)
    db.flush()
    league = models.League(name="t", created_by=user.id, current_week=6, year=2026)
    db.add(league)
    db.flush()

    def pick(entry_id, week, team, survived=None):
        entry = db.query(models.Entry).filter_by(league_id=league.id, entry_id=entry_id).one_or_none()
        if entry is None:
            entry = models.Entry(league_id=league.id, owner_id=user.id, entry_id=entry_id)
            db.add(entry)
            db.flush()
        db.add(models.Pick(entry_id_fk=entry.id, week=week, team=team, survived=survived))
        db.commit()

    yield db, league, pick
    db.close()


def survived(db, models, entry_id, week):
    return (db.query(models.Pick).join(models.Entry, models.Pick.entry_id_fk == models.Entry.id)
            .filter(models.Entry.entry_id == entry_id, models.Pick.week == week).one().survived)


def test_parse_results_final_games_only(mods):
    _, nfl, _ = mods
    raw = {"events": [
        game("Dallas Cowboys", "Tampa Bay Buccaneers", 16, 24),
        game("Green Bay Packers", "Chicago Bears", 20, 20),
        game("Kansas City Chiefs", "Las Vegas Raiders", 7, 0, completed=False),
    ]}
    assert nfl.parse_results(raw) == {"DAL": "lost", "TB": "won", "GB": "tie", "CHI": "tie"}


def test_fills_won_and_lost(mods, league_with_picks):
    models, _, auto = mods
    db, league, pick = league_with_picks
    pick("entry_1", 5, "TB")
    pick("entry_2", 5, "DAL")
    n = auto.fill_pending_results(db, league, results_for=lambda w, y: {"TB": "won", "DAL": "lost"})
    assert n == 2
    assert survived(db, models, "entry_1", 5) is True
    assert survived(db, models, "entry_2", 5) is False


def test_tie_follows_league_rule(mods, league_with_picks):
    models, _, auto = mods
    db, league, pick = league_with_picks
    pick("entry_1", 5, "GB")
    league.tie_counts_as_loss = True
    auto.fill_pending_results(db, league, results_for=lambda w, y: {"GB": "tie"})
    assert survived(db, models, "entry_1", 5) is False

    pick("entry_2", 5, "CHI")
    league.tie_counts_as_loss = False
    auto.fill_pending_results(db, league, results_for=lambda w, y: {"CHI": "tie"})
    assert survived(db, models, "entry_2", 5) is True


def test_game_not_final_stays_pending(mods, league_with_picks):
    models, _, auto = mods
    db, league, pick = league_with_picks
    pick("entry_1", 6, "KC")
    assert auto.fill_pending_results(db, league, results_for=lambda w, y: {}) == 0
    assert survived(db, models, "entry_1", 6) is None


def test_hand_marked_result_is_left_alone(mods, league_with_picks):
    models, _, auto = mods
    db, league, pick = league_with_picks
    pick("entry_1", 5, "DAL", survived=True)  # marked Won by hand, whatever ESPN says
    assert auto.fill_pending_results(db, league, results_for=lambda w, y: {"DAL": "lost"}) == 0
    assert survived(db, models, "entry_1", 5) is True


def test_future_week_is_not_touched(mods, league_with_picks):
    models, _, auto = mods
    db, league, pick = league_with_picks
    pick("entry_1", 7, "KC")  # after the league's current week (6)
    auto.fill_pending_results(db, league, results_for=lambda w, y: {"KC": "won"})
    assert survived(db, models, "entry_1", 7) is None


def test_espn_down_changes_nothing(mods, league_with_picks):
    models, _, auto = mods
    db, league, pick = league_with_picks
    pick("entry_1", 5, "TB")

    def boom(week, year):
        raise ConnectionError("ESPN down")
    assert auto.fill_pending_results(db, league, results_for=boom) == 0
    assert survived(db, models, "entry_1", 5) is None
