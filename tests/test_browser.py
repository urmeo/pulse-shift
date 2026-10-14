import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

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


@pytest.fixture(params=[1280, 390], ids=["desktop", "mobile"])
def page(browser, base_url, request):
    context = browser.new_context(
        viewport={"width": request.param, "height": 900},
        reduced_motion="reduce",
    )
    errors = []
    failed_assets = []
    external_requests = []

    def local_requests_only(route):
        if route.request.url.startswith(base_url):
            route.continue_()
        else:
            external_requests.append(route.request.url)
            route.abort()

    context.route("**/*", local_requests_only)
    page = context.new_page()
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
    try:
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
    finally:
        context.close()


def check_disclosure(details):
    summary = details.locator("summary")
    assert not details.evaluate("element => element.open"), "Disclosure starts open"
    summary.focus()
    expect(summary).to_be_focused()
    summary.press("Enter")
    expect(details).to_have_attribute("open", "")
    summary.press("Space")
    expect(details).not_to_have_attribute("open", "")


def check_pipeline(page):
    image = page.locator("img.pipeline-image, img.pipeline")
    expect(image).to_have_count(1)
    image.evaluate("element => element.loading = 'eager'")
    image.evaluate("element => element.decode()")
    expected = (
        "pipeline-mobile.svg" if page.viewport_size["width"] <= 640 else "pipeline.svg"
    )
    source = image.evaluate("element => element.currentSrc")
    assert urlparse(source).path == f"/results/figures/{expected}", source
    assert image.get_attribute("alt"), "Pipeline needs an accessible description"
    dimensions = image.evaluate(
        """element => {
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return {
                width: rect.width - parseFloat(style.borderLeftWidth)
                    - parseFloat(style.borderRightWidth)
                    - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
                height: rect.height - parseFloat(style.borderTopWidth)
                    - parseFloat(style.borderBottomWidth)
                    - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom),
                naturalWidth: element.naturalWidth,
                naturalHeight: element.naturalHeight
            };
        }"""
    )
    assert dimensions["width"] > 0 and dimensions["height"] > 0, dimensions
    assert dimensions["width"] / dimensions["height"] == pytest.approx(
        dimensions["naturalWidth"] / dimensions["naturalHeight"], rel=0.001
    ), dimensions


def test_manual_estimate_and_pipeline(page, base_url):
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

    check_pipeline(page)
    check_disclosure(page.locator("#pipeline-text"))


def test_saved_results_and_pipeline(page, base_url):
    summary = json.loads((ROOT / "results/tables/summary.json").read_text())
    model = next(
        row
        for row in summary["model_comparison"]
        if row["model"] == "Logistic (unweighted)"
    )
    page.goto(base_url + "results/")
    expect(page.locator(".metrics strong")).to_have_text(
        [
            f"{model['auroc']:.3f}",
            f"{model['auprc']:.3f}",
            f"{model['brier']:.3f}",
            f"{summary['active_hours']:,}",
        ]
    )
    rows = page.locator(".table-scroll tbody tr")
    expect(rows).to_have_count(len(summary["model_comparison"]))
    for row, saved in zip(rows.all(), summary["model_comparison"]):
        expect(row.locator("th, td")).to_have_text(
            [saved["model"]]
            + [f"{saved[key]:.3f}" for key in ("auroc", "auprc", "brier", "ece")]
        )
    check_pipeline(page)
    check_disclosure(page.locator("details"))


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
