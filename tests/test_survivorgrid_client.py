import pandas as pd
import pytest

import survivor.data.survivorgrid_client as survivorgrid_client
from survivor.data.survivorgrid_client import (
    dedupe_schedule_games,
    fetch_season_to_date_games,
    parse_pick_grid,
    parse_schedule_grid,
)

# A minimal fixture mirroring survivorgrid.com's real table structure (verified
# via a live raw-HTML pull): four <td class="dist"> for EV/W%/P%, a
# <td class="teamname"> that may carry a trailing result span, then future
# game cells we don't need. One row uses WSH to check the alias mapping.
SAMPLE_GRID_HTML = """
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th></tr></thead>
  <tbody>
    <tr id="t1">
      <td class="dist">1.14</td>
      <td class="dist">78.0%</td>
      <td class="dist">15.3%</td>
      <td class="teamname">PHI<span class="resultW">&nbsp;(W)</span></td>
    </tr>
    <tr id="t2">
      <td class="dist">0.62</td>
      <td class="dist">62.5%</td>
      <td class="dist">3.1%</td>
      <td class="teamname">KC</td>
    </tr>
    <tr id="t3">
      <td class="dist">0.40</td>
      <td class="dist">55.0%</td>
      <td class="dist">11.1%</td>
      <td class="teamname">WSH</td>
    </tr>
    <tr id="t4">
      <td class="dist">-</td>
      <td class="dist">-</td>
      <td class="dist">-</td>
      <td class="teamname">MIN</td>
    </tr>
    <tr id="t5">
      <td class="dist">0.30</td>
      <td class="dist">22.0%</td>
      <td class="dist">1.0%</td>
      <td class="teamname">DAL<span class="resultL">&nbsp;(L)</span></td>
    </tr>
  </tbody>
</table>
"""


def test_parses_one_row_per_team():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert len(df) == 5
    assert set(df["team"]) == {"PHI", "KC", "WAS", "MIN", "DAL"}


def test_strips_result_marker_from_played_week():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    row = df[df["team"] == "PHI"].iloc[0]
    assert row["win_probability"] == pytest.approx(0.78)
    assert row["pick_percentage"] == pytest.approx(0.153)
    assert row["expected_value"] == pytest.approx(1.14)


def test_alias_team_normalized_to_canonical():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    row = df[df["team"] == "WAS"].iloc[0]
    assert row["pick_percentage"] == pytest.approx(0.111)


def test_missing_values_become_none():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    row = df[df["team"] == "MIN"].iloc[0]
    assert pd.isna(row["win_probability"])
    assert pd.isna(row["pick_percentage"])
    assert pd.isna(row["expected_value"])


def test_pick_percentages_are_plausible_fractions():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    valid = df["pick_percentage"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_missing_table_raises():
    with pytest.raises(ValueError):
        parse_pick_grid("<html><body>no grid here</body></html>")


SCHEDULE_GRID_HTML = """
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th><th>1</th><th>2</th><th>3</th></tr></thead>
  <tbody>
    <tr id="t1">
      <td class="dist">1.07</td>
      <td class="dist">79.8%</td>
      <td class="dist">26.8%</td>
      <td class="teamname">LAC</td>
      <td class="gc g19">
        ARI<br>
        <span class="spread">-9.5</span>
      </td>
      <td class="gc rd">
        @BUF<br>
        <span class="spread">3</span>
      </td>
      <td class="gc bye">BYE</td>
      <td class="fv" data-sort-value="0.47"><div class="starrating"></div></td>
    </tr>
    <tr id="t2">
      <td class="dist">0.80</td>
      <td class="dist">60.0%</td>
      <td class="dist">5.0%</td>
      <td class="teamname">SF</td>
      <td class="gc g6 rd">
        <span style="font-size: 9px;" title="Neutral Field">(n)</span>MIN<br>
        <span class="spread">-3</span>
      </td>
      <td class="gc g2">
        DEN<br>
        <span class="spread">-3.5</span>
      </td>
      <td class="gc rd dv">
        @SEA<br>
        <span class="spread">3</span>
      </td>
      <td class="fv" data-sort-value="0.5"><div class="starrating"></div></td>
    </tr>
  </tbody>
</table>
"""


def test_schedule_grid_parses_home_and_away_with_own_spread():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    lac_week1 = df[(df.team == "LAC") & (df.week == 1)].iloc[0]
    assert lac_week1["opponent"] == "ARI"
    assert lac_week1["is_home"] is True
    assert lac_week1["spread"] == pytest.approx(-9.5)

    lac_week2 = df[(df.team == "LAC") & (df.week == 2)].iloc[0]
    assert lac_week2["opponent"] == "BUF"
    assert lac_week2["is_home"] is False
    assert lac_week2["spread"] == pytest.approx(3.0)


def test_schedule_grid_marks_bye_week():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    lac_week3 = df[(df.team == "LAC") & (df.week == 3)].iloc[0]
    assert bool(lac_week3["is_bye"])
    assert pd.isna(lac_week3["opponent"])
    assert pd.isna(lac_week3["spread"])


FINAL_WEEK_HTML = """
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th><th>18</th></tr></thead>
  <tbody>
    <tr id="t9">
      <td class="dist">1.42</td>
      <td class="dist">84.9%</td>
      <td class="dist">7.5%</td>
      <td class="teamname">DAL<span class="resultW">&nbsp;(W)</span></td>
      <td class="gc rd dv">
        @WAS<br>
        <span class="spread">-13.5</span>
      </td>
    </tr>
  </tbody>
</table>
"""


def test_schedule_grid_handles_season_final_week_with_no_trailing_fv_cell():
    # A season's last week page omits the trailing "fv" (future value) cell
    # entirely, since there's no more season left to rate -- confirmed on
    # the real /2023/18 page, which has exactly 5 cells per row, all real.
    df = parse_schedule_grid(FINAL_WEEK_HTML, start_week=18)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["team"] == "DAL"
    assert row["week"] == 18
    assert row["opponent"] == "WAS"
    assert row["spread"] == pytest.approx(-13.5)


def test_schedule_grid_handles_neutral_site_marker():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    sf_week1 = df[(df.team == "SF") & (df.week == 1)].iloc[0]
    assert sf_week1["opponent"] == "MIN"
    assert sf_week1["is_home"] is False  # "rd" class present despite the (n) marker


def test_schedule_grid_week_numbers_offset_from_start_week():
    df = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=1)
    assert set(df[df.team == "LAC"]["week"]) == {1, 2, 3}

    df_offset = parse_schedule_grid(SCHEDULE_GRID_HTML, start_week=5)
    assert set(df_offset[df_offset.team == "LAC"]["week"]) == {5, 6, 7}


_FV_CELL = '<td class="fv" data-sort-value="0.5"><div class="starrating"></div></td>'

NEUTRAL_GAME_HTML = f"""
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th><th>1</th></tr></thead>
  <tbody>
    <tr><td class="dist">1.0</td><td class="dist">50%</td><td class="dist">5%</td>
        <td class="teamname">BUF</td>
        <td class="gc">NE<br><span class="spread">-3</span></td>
        {_FV_CELL}</tr>
    <tr><td class="dist">1.0</td><td class="dist">50%</td><td class="dist">5%</td>
        <td class="teamname">NE</td>
        <td class="gc rd">@BUF<br><span class="spread">3</span></td>
        {_FV_CELL}</tr>
    <tr><td class="dist">1.0</td><td class="dist">40%</td><td class="dist">2%</td>
        <td class="teamname">LAR</td>
        <td class="gc rd">
          <span title="Neutral Field">(n)</span>SF<br>
          <span class="spread">-3.5</span>
        </td>
        {_FV_CELL}</tr>
    <tr><td class="dist">1.0</td><td class="dist">60%</td><td class="dist">10%</td>
        <td class="teamname">SF</td>
        <td class="gc rd">
          <span title="Neutral Field">(n)</span>LAR<br>
          <span class="spread">3.5</span>
        </td>
        {_FV_CELL}</tr>
    <tr><td class="dist">-</td><td class="dist">-</td><td class="dist">-</td>
        <td class="teamname">KC</td>
        <td class="gc bye">BYE</td>
        {_FV_CELL}</tr>
  </tbody>
</table>
"""


def test_dedupe_keeps_one_row_for_a_normal_home_away_game():
    grid = parse_schedule_grid(NEUTRAL_GAME_HTML, start_week=1)
    games = dedupe_schedule_games(grid)
    buf_ne = games[(games.home_team == "BUF") & (games.away_team == "NE")]
    assert len(buf_ne) == 1
    assert buf_ne.iloc[0]["home_spread"] == pytest.approx(-3.0)


def test_dedupe_recovers_neutral_site_game_marked_away_on_both_sides():
    grid = parse_schedule_grid(NEUTRAL_GAME_HTML, start_week=1)
    games = dedupe_schedule_games(grid)
    neutral_game = games[
        ((games.home_team == "LAR") & (games.away_team == "SF"))
        | ((games.home_team == "SF") & (games.away_team == "LAR"))
    ]
    assert len(neutral_game) == 1  # not dropped, and not duplicated


def test_dedupe_excludes_bye_weeks():
    grid = parse_schedule_grid(NEUTRAL_GAME_HTML, start_week=1)
    games = dedupe_schedule_games(grid)
    assert not (games.home_team == "KC").any()
    assert not (games.away_team == "KC").any()


def test_dedupe_total_game_count():
    grid = parse_schedule_grid(NEUTRAL_GAME_HTML, start_week=1)
    games = dedupe_schedule_games(grid)
    assert len(games) == 2  # BUF/NE and LAR/SF -- KC's bye contributes nothing


def test_extracts_win_result():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert df[df["team"] == "PHI"].iloc[0]["result"] == "W"


def test_extracts_loss_result():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert df[df["team"] == "DAL"].iloc[0]["result"] == "L"


def test_upcoming_week_has_no_result():
    df = parse_pick_grid(SAMPLE_GRID_HTML)
    assert pd.isna(df[df["team"] == "KC"].iloc[0]["result"])


def _one_column_page(cell_html: str, cell_class: str = "gc") -> str:
    # A minimal single-gc-column page: parse_schedule_grid(html, start_week=W)
    # will tag this one column's game as week W, whatever W is passed for
    # that call -- exactly what fetch_season_to_date_games relies on (each
    # week's own page has *its own* game at offset 0). is_home comes from
    # cell_class containing "rd", not from an "@" in cell_html -- that's
    # just a display marker, per parse_schedule_grid's real behavior.
    return f"""
    <table class="datatable" id="grid">
      <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th><th>1</th></tr></thead>
      <tbody>
        <tr><td class="dist">1.0</td><td class="dist">50%</td><td class="dist">5%</td>
            <td class="teamname">LAC</td>
            <td class="{cell_class}">{cell_html}</td>
            {_FV_CELL}</tr>
      </tbody>
    </table>
    """


_BYE_PAGE = f"""
<table class="datatable" id="grid">
  <thead><tr><th>EV</th><th>W%</th><th>P%</th><th>Team</th><th>1</th></tr></thead>
  <tbody>
    <tr><td class="dist">-</td><td class="dist">-</td><td class="dist">-</td>
        <td class="teamname">LAC</td>
        <td class="gc bye">BYE</td>
        {_FV_CELL}</tr>
  </tbody>
</table>
"""

WEEK_PAGES = {
    1: _one_column_page('ARI<br><span class="spread">-9.5</span>'),  # LAC home vs ARI
    2: _one_column_page('@BUF<br><span class="spread">3</span>', cell_class="gc rd"),  # LAC away at BUF
    3: _BYE_PAGE,
}


def test_fetch_season_to_date_games_fetches_one_page_per_week(monkeypatch):
    calls = []

    def fake_fetch_week_html(year, week):
        calls.append((year, week))
        return WEEK_PAGES[week]

    monkeypatch.setattr(survivorgrid_client, "fetch_week_html", fake_fetch_week_html)
    monkeypatch.setattr(survivorgrid_client.time, "sleep", lambda _: None)

    fetch_season_to_date_games(2026, through_week=3, delay_seconds=0, cache_dir=None)
    assert calls == [(2026, 1), (2026, 2), (2026, 3)]


def test_fetch_season_to_date_games_tags_each_row_with_its_own_week(monkeypatch):
    # No counterpart row is provided for either fixture (a single team's
    # page in isolation), so dedupe_schedule_games' fallback always labels
    # that row's own team as "home" -- home/away preference itself is
    # covered by dedupe_schedule_games' own tests. What this test actually
    # verifies is that each week's own page contributes its own opponent,
    # not week1's data relabeled for every week.
    monkeypatch.setattr(survivorgrid_client, "fetch_week_html", lambda year, week: WEEK_PAGES[week])
    monkeypatch.setattr(survivorgrid_client.time, "sleep", lambda _: None)

    games = fetch_season_to_date_games(2026, through_week=3, delay_seconds=0, cache_dir=None)
    assert set(games["week"]) == {1, 2}  # week 3 is a bye, dropped -- see next test
    week1 = games[games.week == 1].iloc[0]
    assert week1["home_team"] == "LAC" and week1["away_team"] == "ARI"
    week2 = games[games.week == 2].iloc[0]
    assert week2["home_team"] == "LAC" and week2["away_team"] == "BUF"


def test_fetch_season_to_date_games_drops_bye_weeks(monkeypatch):
    monkeypatch.setattr(survivorgrid_client, "fetch_week_html", lambda year, week: WEEK_PAGES[week])
    monkeypatch.setattr(survivorgrid_client.time, "sleep", lambda _: None)

    games = fetch_season_to_date_games(2026, through_week=3, delay_seconds=0, cache_dir=None)
    assert not (games["week"] == 3).any()


def test_fetch_week_html_cached_fetches_once_then_serves_from_disk(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(survivorgrid_client, "fetch_week_html", lambda year, week: calls.append((year, week)) or "<html>page</html>")

    first = survivorgrid_client.fetch_week_html_cached(2025, 3, tmp_path)
    second = survivorgrid_client.fetch_week_html_cached(2025, 3, tmp_path)

    assert first == ("<html>page</html>", False)
    assert second == ("<html>page</html>", True)
    assert calls == [(2025, 3)]
    assert not list(tmp_path.glob("*.tmp"))


def test_fetch_week_html_cached_with_no_cache_dir_always_fetches_and_writes_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(survivorgrid_client, "fetch_week_html", lambda year, week: calls.append(1) or "<html/>")

    survivorgrid_client.fetch_week_html_cached(2025, 3, None)
    survivorgrid_client.fetch_week_html_cached(2025, 3, None)

    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


def test_season_to_date_games_only_sleeps_after_live_fetches(tmp_path, monkeypatch):
    games = pd.DataFrame({"week": [1], "home_team": ["BUF"], "away_team": ["NYJ"], "home_spread": [-3.0]})
    monkeypatch.setattr(survivorgrid_client, "fetch_week_html", lambda year, week: "<html/>")
    monkeypatch.setattr(survivorgrid_client, "parse_schedule_grid", lambda html, start_week: None)
    monkeypatch.setattr(survivorgrid_client, "dedupe_schedule_games", lambda grid: games)
    sleeps = []
    monkeypatch.setattr(survivorgrid_client.time, "sleep", sleeps.append)

    survivorgrid_client.fetch_season_to_date_games(2026, through_week=1, delay_seconds=1.0, cache_dir=tmp_path)
    assert sleeps == [1.0]

    # second run is served from the cache: no fetch, so no courtesy delay
    survivorgrid_client.fetch_season_to_date_games(2026, through_week=1, delay_seconds=1.0, cache_dir=tmp_path)
    assert sleeps == [1.0]
