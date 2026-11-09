"""Feature execution choices derived from settings."""

from mlxradiomics.config.settings import Settings


def explicit_texture(settings: Settings) -> bool:
    """Whether texture must use the explicit matrix pipeline instead of the native one.

    True when any family has a non-``native`` aggregation, neighborhood support is
    ``closed-radius``, NGTDM singular cases use ``ibsi-caps``, or a GLDZM feature is selected.
    """
    return (
        settings.neighborhood_support == "closed-radius"
        or settings.ngtdm_singular_cases == "ibsi-caps"
        or any(
            mode != "native"
            for mode in (
                settings.glcm_aggregation,
                settings.glrlm_aggregation,
                settings.glszm_aggregation,
                settings.gldzm_aggregation,
                settings.ngtdm_aggregation,
                settings.gldm_aggregation,
            )
        )
        or any(name.startswith("gldzm_") for name in settings.features)
    )
