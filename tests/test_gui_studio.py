"""Regression checks for the refreshed topology editor UI."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from gui.studio_styles import _mix
from gui.theme import THEMES, ViewSettings


class StudioThemeTests(unittest.TestCase):
    def test_blend_extremes(self):
        self.assertEqual(_mix("#123456", "#abcdef", 0.0), "#123456")
        self.assertEqual(_mix("#123456", "#abcdef", 1.0), "#abcdef")

    def test_palettes_have_required_colors(self):
        keys = ("panel_bg", "panel_fg", "field_bg", "hint_fg", "button_bg",
                "grid_major", "accent", "canvas_bg")
        for name, palette in THEMES.items():
            with self.subTest(theme=name):
                for key in keys:
                    self.assertRegex(palette[key], r"^#[0-9a-fA-F]{6}$")

    def test_view_preferences_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "gui_view.json")
            view = ViewSettings(theme="light", panel_width=410, panel_visible=False)
            view.save(path)
            restored = ViewSettings.load(path)
            self.assertEqual(restored.theme, "light")
            self.assertEqual(restored.panel_width, 410)
            self.assertFalse(restored.panel_visible)


@unittest.skipUnless(os.environ.get("DISPLAY") or sys.platform in ("win32", "darwin"),
                     "GUI smoke check requires a graphical display (use xvfb-run)")
class StudioIntegrationTests(unittest.TestCase):
    def test_editor_commands_and_inspector(self):
        import tkinter as tk
        from gui.editor import TopologyEditor

        root = tk.Tk()
        try:
            app = TopologyEditor(root)
            root.update()
            self.assertTrue(app.prop_scroll.content.winfo_exists())
            self.assertEqual(app.mode.get(), "select")
            app.choose_mode("link")
            self.assertEqual(app.mode.get(), "link")
            app.cancel_action()
            self.assertEqual(app.mode.get(), "select")
            app.zoom_by(1.2)
            self.assertGreater(app.scale, 0)
            app.toggle_panel()
            self.assertNotIn(str(app.right), app.split.panes())
            app.toggle_panel()
            self.assertIn(str(app.right), app.split.panes())
            app.set_theme("midnight")
            self.assertEqual(app.view.theme, "midnight")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
