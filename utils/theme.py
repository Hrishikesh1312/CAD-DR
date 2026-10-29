"""Session-scoped colors for the workspace and Plotly charts."""

PALETTES = {
    "Light": {
        "background": "#F8FAFC", "surface": "#FFFFFF", "sidebar": "#EEF2F6",
        "text": "#172B4D", "muted": "#52637B", "border": "#DCE3ED",
        "accent": "#2563EB", "focus": "#93B4FA", "mesh": "#6B91BC",
        "voxel": "#3F72AF", "button-text": "#FFFFFF",
    },
    "Dark": {
        "background": "#1C1C1E", "surface": "#272729", "sidebar": "#222224",
        "text": "#EEEEF0", "muted": "#B4B4BA", "border": "#414145",
        "accent": "#79B8FF", "focus": "#B5D8FF", "mesh": "#8BB7E8",
        "voxel": "#78A7E3", "button-text": "#1C1C1E",
    },
}


def get_palette(theme):
    return PALETTES.get(theme, PALETTES["Dark"])


def theme_variables(theme):
    palette = get_palette(theme)
    variables = ";".join(f"--cad-{name}:{value}" for name, value in palette.items())
    scheme = "dark" if theme == "Dark" else "light"
    return f"<style>.stApp {{{variables}; color-scheme:{scheme};}}</style>"
