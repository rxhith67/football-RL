"""Headless local browser verification, with actual published experiment data."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    target = ROOT / "artifacts/browser-check"
    target.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(ROOT / "line_coach_demo")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result = {"success": False, "layout_only": args.layout_only}
    errors = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1100}, device_scale_factor=1,
                                          record_video_dir=str(target / "video") if not args.layout_only else None)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
            if not args.layout_only:
                page.wait_for_function("document.getElementById('play').disabled === false")
                assert not page.locator("#error").is_visible()
                assert page.locator("#results-body tr").count() == 7
                assert page.locator("#policy option").count() == 7
                page.get_by_role("button", name="Play replay").click()
                page.wait_for_function("Number(document.getElementById('scrub').value) > 2")
                page.get_by_role("button", name="Pause replay").click()
                page.get_by_role("button", name="Next decision").click()
                assert int(page.locator("#scrub").input_value()) >= 20
                page.locator("#scrub").evaluate("element => { element.value = '600'; element.dispatchEvent(new Event('input', {bubbles:true})); }")
                assert "600 /" in page.locator("#step").inner_text()
                page.get_by_role("button", name="Restart").click()
                assert page.locator("#scrub").input_value() == "0"
                page.locator("#policy").select_option(label="Always high")
                page.wait_for_function("document.getElementById('controller').textContent === 'high'")
                page.get_by_role("button", name="Next decision").click()
                assert page.locator("#instruction").inner_text() == "High line"
                page.locator("#speed").select_option("8")
                assert page.locator("#speed").input_value() == "8"
                result["controls_checked"] = ["play", "pause", "restart", "decision jump", "scrub", "policy selection", "speed"]
                assert page.locator("#results").inner_text().find("goal difference") >= 0
                result["result_rows"] = 7
                # Record an actual complete replay as a presentation backup.
                page.locator("#policy").select_option(index=0)
                page.wait_for_function("document.getElementById('controller').textContent.startsWith('PPO')")
                page.get_by_role("button", name="Play replay").click()
                page.wait_for_function("document.getElementById('scrub').value === document.getElementById('scrub').max", timeout=45000)
                page.get_by_role("button", name="Play replay", exact=True).wait_for(state="visible", timeout=45000)
                result["complete_replay_played"] = True
                page.locator("#scrub").evaluate("element => { element.value = '600'; element.dispatchEvent(new Event('input', {bubbles:true})); }")
            page.screenshot(path=str(target / "desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(target / "mobile.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "Mobile overflow"
            assert not errors, errors
            result.update(success=True, javascript_errors=errors, mobile_overflow=False)
            video = page.video
            context.close()
            if video is not None:
                result["video"] = str(Path(video.path()).relative_to(ROOT))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        (target / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
