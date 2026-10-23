from contextlib import contextmanager
from functools import partial
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urljoin

import pytest
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base_url():
    handler = partial(QuietHandler, directory=str(ROOT))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/"
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel="chrome")
        try:
            yield browser
        finally:
            browser.close()


@contextmanager
def local_page(browser, base_url, blocked=None, **options):
    with browser.new_context(reduced_motion="reduce", **options) as context:

        def local_requests_only(route):
            if route.request.url.startswith(base_url):
                route.continue_()
            else:
                if blocked is not None:
                    blocked.append(route.request.url)
                route.abort()

        context.route("**/*", local_requests_only)
        yield context.new_page()


@pytest.fixture(params=[1280, 390], ids=["desktop", "mobile"])
def page(browser, base_url, request):
    errors, failed_assets, external_requests = [], [], []
    with local_page(
        browser,
        base_url,
        blocked=external_requests,
        viewport={"width": request.param, "height": 900},
    ) as page:
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "response",
            lambda response: (
                failed_assets.append(f"{response.status} {response.url}")
                if response.url.startswith(base_url) and response.status >= 400
                else None
            ),
        )
        page.on(
            "requestfailed",
            lambda request: (
                failed_assets.append(f"{request.failure} {request.url}")
                if request.url.startswith(base_url)
                else None
            ),
        )
        yield page
        page.locator("img").evaluate_all(
            "images => images.forEach(image => image.loading = 'eager')"
        )
        page.wait_for_function(
            """() => [...document.images].every(
                image => image.complete && image.naturalWidth > 0
            )"""
        )
        page.locator("img").evaluate_all(
            "images => Promise.all(images.map(image => image.decode()))"
        )
        sizes = page.evaluate(
            """() => ({
                viewport: innerWidth,
                document: document.documentElement.scrollWidth,
                body: document.body.scrollWidth
            })"""
        )
        assert sizes["document"] <= sizes["viewport"], sizes
        assert sizes["body"] <= sizes["viewport"], sizes
        assert not errors, f"Browser errors: {errors}"
        assert not failed_assets, f"Failed local assets: {failed_assets}"
        assert not external_requests, (
            f"Unexpected external requests: {external_requests}"
        )


def check_disclosure(details):
    summary = details.locator("summary")
    assert not details.evaluate("element => element.open"), "Disclosure starts open"
    summary.focus()
    expect(summary).to_be_focused()
    summary.press("Enter")
    expect(details).to_have_attribute("open", "")
    summary.press("Space")
    expect(details).not_to_have_attribute("open", "")


@pytest.mark.parametrize("route", ["", "index.html", "app/", "app/index.html"])
def test_app_routes_preserve_bookmarks(page, base_url, route):
    bookmark = "?source=bookmark#form"
    page.goto(urljoin(base_url, route) + bookmark)
    destination = "app/" if route in ("", "index.html") else route
    expect(page).to_have_url(urljoin(base_url, destination) + bookmark)
    expect(page.locator("#result")).to_be_visible()
    expect(page.locator("#form")).to_be_in_viewport()


def test_root_link_without_javascript(browser, page, base_url):
    with local_page(
        browser,
        base_url,
        viewport=page.viewport_size,
        java_script_enabled=False,
    ) as fallback_page:
        fallback_page.goto(base_url)
        link = fallback_page.locator('a[href="app/"]')
        expect(link).to_be_visible()
        link.click()
        expect(fallback_page).to_have_url(urljoin(base_url, "app/"))
        expect(fallback_page.locator("noscript p")).to_be_visible()
        expect(fallback_page.locator("noscript p")).to_have_text(
            "Enable JavaScript to use the demand model."
        )
        expect(fallback_page.locator("#result")).to_be_hidden()


def test_manual_estimate(page, base_url):
    page.goto(base_url)
    result = page.locator("#result")
    score = page.locator("#pct")
    expect(result).to_be_visible()
    page.locator("#hour").select_option("17")
    original_score = score.inner_text()
    assert original_score.endswith("%"), original_score

    precipitation = page.locator("#precip")
    precipitation.fill("0.4")
    expect(score).not_to_have_text(original_score)
    precipitation.fill("")
    expect(result).to_be_hidden()
    expect(precipitation).to_have_attribute("aria-invalid", "true")
    expect(page.locator("#error")).to_contain_text("Precipitation")
    precipitation.fill("0")
    expect(result).to_be_visible()
    expect(score).to_have_text(original_score)
    expect(page.locator("#error")).to_be_empty()

    expect(
        page.locator('#pipeline, a[href="#pipeline"], img[src*="pipeline"]')
    ).to_have_count(0)


def test_forecast_load_submit_and_manual_edit(page, base_url):
    page.add_init_script("Date.now = () => Date.parse('2026-10-10T14:20:00Z');")
    data = json.loads((ROOT / "tests/fixtures/forecast.json").read_text())

    def serve_forecast(route):
        source = "air" if "air-quality-api" in route.request.url else "weather"
        route.fulfill(
            json=data[source],
            headers={"Access-Control-Allow-Origin": base_url.rstrip("/")},
        )

    for url in [
        "https://api.open-meteo.com/v1/forecast?*",
        "https://air-quality-api.open-meteo.com/v1/air-quality?*",
    ]:
        page.route(url, serve_forecast)
    page.goto(base_url)
    expect(page.locator("#result")).to_be_visible()
    page.locator("#smoke").check()
    button = page.locator("#live")
    button.click()
    expect(button).to_have_text("DC forecast loaded")
    expect(button).to_be_enabled()
    expect(button).not_to_have_attribute("aria-busy", "true")
    for field, value in {
        "temp": "75",
        "humidity": "60",
        "aqi": "42",
        "wind": "0",
        "precip": "0.24",
        "visibility": "3",
        "hour": "10",
    }.items():
        expect(page.locator(f"#{field}")).to_have_value(value)
    expect(page.locator("#weekend")).to_be_checked()
    expect(page.locator("#smoke")).to_be_checked()
    expect(page.locator("#error")).to_be_empty()
    best_hour = page.locator("#besthour")
    forecast_note = page.locator("#forecastnote")
    expect(best_hour).to_contain_text("Lowest estimated chance")
    expect(best_hour).to_contain_text("Oct 10")
    expect(best_hour).to_contain_text("EDT")
    expect(best_hour).to_contain_text("11:00")
    expect(best_hour).to_contain_text("AQI 70")
    expect(forecast_note).to_contain_text("Smoke/haze uses your checkbox")
    original_ranking = best_hour.inner_text()
    original_note = forecast_note.inner_text()
    page.locator("#temp").press("Enter")
    page.locator("#form").evaluate("form => form.requestSubmit()")
    expect(best_hour).to_have_text(original_ranking)
    expect(forecast_note).to_have_text(original_note)
    expect(button).to_have_text("DC forecast loaded")
    assert page.url == urljoin(base_url, "app/")
    page.locator("#temp").fill("76")
    expect(best_hour).to_be_empty()
    expect(forecast_note).to_be_empty()
    expect(button).to_have_text("Load DC forecast")


def test_malformed_model_shows_startup_error(page, base_url):
    page.route(
        "**/app/model.js",
        lambda route: route.fulfill(
            content_type="text/javascript", body="window.PULSESHIFT_MODEL = {};"
        ),
    )
    page.goto(base_url)
    expect(page.locator("#error")).to_be_visible()
    expect(page.locator("#error")).to_have_text(
        "The demand model could not be loaded. Reload the page."
    )
    expect(page.locator("#live")).to_be_disabled()
    expect(page.locator("#result")).to_be_hidden()


def test_unavailable_model_module_shows_startup_error(browser, base_url):
    with local_page(browser, base_url, viewport={"width": 1280, "height": 900}) as page:
        page.route(
            "**/app/demand-model.mjs",
            lambda route: route.fulfill(status=404, body="Module unavailable"),
        )
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url)
        expect(page.locator("#error")).to_be_visible()
        expect(page.locator("#error")).to_have_text(
            "The demand model could not be loaded. Reload the page."
        )
        expect(page.locator("#live")).to_be_disabled()
        expect(page.locator("#result")).to_be_hidden()
        assert not errors, f"Uncaught browser errors: {errors}"


def test_walkthrough_media_and_transcripts(page, base_url):
    page.goto(base_url + "docs/demos/")
    videos = page.locator("video")
    expect(videos).to_have_count(2)
    page.wait_for_function(
        """() => [...document.querySelectorAll('video')].every(video =>
            video.readyState >= 1 && Number.isFinite(video.duration)
            && video.duration > 0 && video.videoWidth > 0
        )"""
    )
    for video in videos.all():
        assert video.evaluate("element => element.controls"), "Video needs controls"
        poster = video.evaluate(
            """async video => {
                const image = new Image();
                image.src = video.poster;
                await image.decode();
                return {width: image.naturalWidth, height: image.naturalHeight};
            }"""
        )
        assert poster["width"] > 0 and poster["height"] > 0, poster

        track = video.locator("track[kind='captions']")
        expect(track).to_have_count(1)
        expect(track).to_have_attribute("srclang", "en")
        video.evaluate("element => element.textTracks[0].mode = 'hidden'")
        page.wait_for_function(
            "element => element.readyState === 2", arg=track.element_handle()
        )
        assert track.evaluate("element => element.track.cues.length") > 0, (
            "Captions are empty"
        )

        transcript_id = video.get_attribute("aria-describedby")
        assert transcript_id, "Video needs a transcript link"
        transcript = page.locator(f"#{transcript_id}")
        assert transcript.locator("li").count() > 0, "Transcript is empty"
        check_disclosure(transcript)
