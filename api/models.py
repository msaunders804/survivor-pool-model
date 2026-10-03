"""Data model for Phase 9.

Five tables, matching the build order in hosting_plan.md:
users, leagues, league_memberships, entries, recommendations.
`picks` is the sixth -- the fast-follow -- shaped to match what
survivor.data.my_entries already writes to CSV, so ingesting Megan's
weekly artifact is a straight row-for-row load, not a transform.

SQLite locally (one file, zero setup). Point DATABASE_URL at Postgres
on Render and nothing else here changes -- SQLAlchemy handles both.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Absolute path, anchored to this file's own directory -- not the current
# working directory. A relative "./survivor.db" would point at a different
# file depending on whether something ran from api/ or the repo root
# (e.g. the weekly job's own process vs. a subprocess it launches), which
# silently looks like an empty, table-less database rather than an error
# pointing at the real one. On Render this is overridden by DATABASE_URL
# pointing at Postgres, so this default only matters for local dev.
_LOCAL_DB_PATH = Path(__file__).resolve().parent / "survivor.db"
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_LOCAL_DB_PATH}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=now)

    memberships = relationship("LeagueMembership", back_populates="user")


class League(Base):
    __tablename__ = "leagues"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=now)

    # v1 config -- the five her hosting_plan names (replaces the hardcoded
    # constants in portfolio.py / run_weekly.py's CLI defaults)
    entry_cap = Column(Integer, default=10)
    buy_in = Column(Float, default=20.0)
    rake_pct = Column(Float, default=0.10)
    pot = Column(Float, default=9000.0)
    week_start = Column(Integer, default=1)
    week_end = Column(Integer, default=18)

    # not in her v1 config list -- fixed, not exposed, per the wireframe notes
    tie_counts_as_loss = Column(Boolean, default=True)
    pot_rolls_over_if_all_out = Column(Boolean, default=False)

    year = Column(Integer, default=2026)
    n_rivals = Column(Integer, default=500)  # updated from the real entry count near lock
    current_week = Column(Integer, default=4)  # bumped by hand once a week turns over -- see update.html

    memberships = relationship("LeagueMembership", back_populates="league")
    entries = relationship("Entry", back_populates="league")
    recommendations = relationship("Recommendation", back_populates="league")


class LeagueMembership(Base):
    __tablename__ = "league_memberships"
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_creator = Column(Boolean, default=False)  # only the creator edits league config
    __table_args__ = (UniqueConstraint("league_id", "user_id", name="one_membership_per_user_per_league"),)

    league = relationship("League", back_populates="memberships")
    user = relationship("User", back_populates="memberships")


class Entry(Base):
    """One ticket. entry_id ('entry_1', 'BMJ1', ...) matches what
    survivor.data.my_entries uses as its key, so an entry here and an
    entry in her CSV are the same row by that string."""
    __tablename__ = "entries"
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # private to this user within the league
    entry_id = Column(String, nullable=False)  # e.g. "BMJ1"
    __table_args__ = (UniqueConstraint("league_id", "entry_id", name="unique_entry_id_per_league"),)

    league = relationship("League", back_populates="entries")
    picks = relationship("Pick", back_populates="entry")


class Pick(Base):
    """Mirrors survivor.data.my_entries.PICKS_FILE's (entry_id, week, team,
    survived) exactly. `confirmed` is the one field her CSV doesn't have --
    it's the "on Splash yet" state the wireframe's picks list needs."""
    __tablename__ = "picks"
    id = Column(Integer, primary_key=True)
    entry_id_fk = Column(Integer, ForeignKey("entries.id"), nullable=False)
    week = Column(Integer, nullable=False)
    team = Column(String, nullable=False)
    survived = Column(Boolean, nullable=True)  # NULL = not yet resolved
    confirmed = Column(Boolean, default=False)  # True once entered on Splash
    updated_at = Column(DateTime, default=now, onupdate=now)
    __table_args__ = (UniqueConstraint("entry_id_fk", "week", name="one_pick_per_entry_per_week"),)

    entry = relationship("Entry", back_populates="picks")


class Recommendation(Base):
    """Written only by the weekly job, read-only to everything else --
    the separation that makes 'reset to the model' and an end-of-season
    override comparison possible. One row per (league, week, entry)."""
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    week = Column(Integer, nullable=False)
    entry_id = Column(String, nullable=False)  # string, not FK -- matches her pick_sheets CSV directly
    team = Column(String, nullable=False)
    used_teams_before = Column(String, default="")  # comma-joined, as her CSV writes it
    expected_total_payout = Column(Float, nullable=True)  # same for every row in one run
    payout_se = Column(Float, nullable=True)
    generated_at = Column(DateTime, default=now)
    __table_args__ = (UniqueConstraint("league_id", "week", "entry_id", name="one_recommendation_per_entry_per_week"),)

    league = relationship("League", back_populates="recommendations")


class Ownership(Base):
    """What Brent pastes from Splash's Stats tab: how many of the league
    picked each team, a given week. Separate from Recommendation (the
    model's output) and Pick (your own entries) -- this is the field's
    real behavior, reported, not projected."""
    __tablename__ = "ownership"
    id = Column(Integer, primary_key=True)
    league_id = Column(Integer, ForeignKey("leagues.id"), nullable=False)
    week = Column(Integer, nullable=False)
    team = Column(String, nullable=False)
    pick_count = Column(Integer, nullable=True)   # raw count, when Splash gives one
    pct = Column(Float, nullable=True)             # percentage, when that's what's pasted
    updated_at = Column(DateTime, default=now, onupdate=now)
    __table_args__ = (UniqueConstraint("league_id", "week", "team", name="one_ownership_row_per_team_per_week"),)


def init_db() -> None:
    Base.metadata.create_all(engine)
