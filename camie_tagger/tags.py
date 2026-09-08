"""Conversion of model predictions into hierarchical Immich tags."""

from __future__ import annotations

# meta and year are deliberately excluded: the model's year predictions are unreliable
# and meta tags describe the file rather than its content.
CATEGORIES_TO_WRITE = ("character", "copyright", "artist", "general", "rating")

# Top-level tag namespaces owned by this tool.
CAMIE_ROOTS = frozenset(CATEGORIES_TO_WRITE)

TAG_PREFIXES = tuple(f"{category}/" for category in CATEGORIES_TO_WRITE)


def normalize_tag(name: str) -> str:
    # A slash inside a tag would create an unintended level in the Immich tag tree.
    return name.replace("/", "_")


def build_taglist(
    prediction: dict[str, list[tuple[str, float]]],
    categories: tuple[str, ...] = CATEGORIES_TO_WRITE,
) -> list[str]:
    """Flatten a prediction into deduplicated 'category/tag' strings."""
    taglist: list[str] = []
    seen: set[str] = set()

    for category in categories:
        for name, _score in prediction.get(category, []):
            if category == "rating":
                name = name.removeprefix("rating_")
            tag = f"{category}/{normalize_tag(name)}"
            if tag not in seen:
                seen.add(tag)
                taglist.append(tag)

    return taglist


def has_camie_tags(taglist: list[str]) -> bool:
    return any(tag.startswith(TAG_PREFIXES) for tag in taglist)


def tags_in_namespace(taglist: list[str], namespace: str) -> list[str]:
    prefix = f"{namespace}/"
    return [tag for tag in taglist if tag.startswith(prefix)]
