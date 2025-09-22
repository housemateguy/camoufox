import xml.etree.ElementTree as ET  # nosec
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from random import random
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
from language_tags import tags

from camoufox.pkgman import LOCAL_DATA, GitHubDownloader, rprint, webdl
from camoufox.warnings import LeakWarning

from .exceptions import (
    InvalidLocale,
    MissingRelease,
    NotInstalledGeoIPExtra,
    UnknownIPLocation,
    UnknownLanguage,
    UnknownTerritory,
)
from .ip import validate_ip

try:
    import geoip2.database  # type: ignore
except ImportError:
    ALLOW_GEOIP = False
else:
    ALLOW_GEOIP = True


"""
Data structures for locale and geolocation info
"""


@dataclass(slots=True)
class Locale:
    """
    Stores locale, region, and script information.
    """

    language: str
    region: Optional[str] = None
    script: Optional[str] = None

    @property
    def as_string(self) -> str:
        if self.region:
            return f"{self.language}-{self.region}"
        return self.language

    def as_config(self) -> Dict[str, str]:
        """
        Converts the locale to a intl config dictionary.
        """
        assert self.region
        data = {
            'locale:region': self.region,
            'locale:language': self.language,
        }
        if self.script:
            data['locale:script'] = self.script
        return data


@dataclass(frozen=True, slots=True)
class Geolocation:
    """
    Stores geolocation information.
    """

    locale: Locale
    longitude: float
    latitude: float
    timezone: str
    accuracy: Optional[float] = None

    def as_config(self) -> Dict[str, Any]:
        """
        Converts the geolocation to a config dictionary.
        """
        data = {
            'geolocation:longitude': self.longitude,
            'geolocation:latitude': self.latitude,
            'timezone': self.timezone,
            **self.locale.as_config(),
        }
        if self.accuracy:
            data['geolocation:accuracy'] = self.accuracy
        return data


class ProbabilityTable(NamedTuple):
    values: Tuple[str, ...]
    cumulative: Tuple[float, ...]


def _sample_probability(table: ProbabilityTable) -> str:
    """Select a value from a cumulative distribution."""

    idx = bisect_left(table.cumulative, random())
    if idx >= len(table.values):
        return table.values[-1]
    return table.values[idx]


"""
Helpers to validate and normalize locales
"""


def verify_locale(loc: str) -> None:
    """
    Verifies that a locale is valid.
    Takes either language-region or language.
    """
    if tags.check(loc):
        return
    raise InvalidLocale.invalid_input(loc)


def normalize_locale(locale: str) -> Locale:
    """
    Normalizes and validates a locale code.
    """
    verify_locale(locale)

    # Parse the locale
    parser = tags.tag(locale)
    if not parser.region:
        raise InvalidLocale.invalid_input(locale)

    record = parser.language.data['record']

    # Return a formatted locale object
    return Locale(
        language=record['Subtag'],
        region=parser.region.data['record']['Subtag'],
        script=record.get('Suppress-Script'),
    )


def handle_locale(locale: str, ignore_region: bool = False) -> Locale:
    """
    Handles a locale input, normalizing it if necessary.
    """
    # If the user passed in `language-region` or `language-script-region`, normalize it.
    if len(locale) > 3:
        return normalize_locale(locale)

    # Case: user passed in `region` and needs a full locale
    try:
        return SELECTOR.from_region(locale)
    except UnknownTerritory:
        pass

    # Case: user passed in `language`, and doesn't care about the region
    if ignore_region:
        verify_locale(locale)
        return Locale(language=locale)

    # Case: user passed in `language` and wants a region
    try:
        language = SELECTOR.from_language(locale)
    except UnknownLanguage:
        pass
    else:
        LeakWarning.warn('no_region')
        return language

    # Locale is not in a valid format.
    raise InvalidLocale.invalid_input(locale)


def handle_locales(locales: Union[str, List[str]], config: Dict[str, Any]) -> None:
    """
    Handles a list of locales.
    """
    if isinstance(locales, str):
        locales = [loc.strip() for loc in locales.split(',')]

    # First, handle the first locale. This will be used for the intl api.
    intl_locale = handle_locale(locales[0])
    config.update(intl_locale.as_config())

    if len(locales) < 2:
        return

    # If additional locales were passed, validate them.
    # Note: in this case, we do not need the region.
    config['locale:all'] = _join_unique(
        handle_locale(locale, ignore_region=True).as_string for locale in locales
    )


def _join_unique(seq: Iterable[str]) -> str:
    """
    Joins a sequence of strings without duplicates
    """
    seen: Set[str] = set()
    return ', '.join(x for x in seq if not (x in seen or seen.add(x)))


"""
Helpers to fetch geolocation, timezone, and locale data given an IP.
"""

MMDB_FILE = LOCAL_DATA / 'GeoLite2-City.mmdb'
MMDB_REPO = "P3TERX/GeoLite.mmdb"


class MaxMindDownloader(GitHubDownloader):
    """
    MaxMind database downloader from a GitHub repository.
    """

    def check_asset(self, asset: Dict) -> Optional[str]:
        # Check for the first -City.mmdb file
        if asset['name'].endswith('-City.mmdb'):
            return asset['browser_download_url']
        return None

    def missing_asset_error(self) -> None:
        raise MissingRelease('Failed to find GeoIP database release asset')


def geoip_allowed() -> None:
    """
    Checks if the geoip2 module is available.
    """
    if not ALLOW_GEOIP:
        raise NotInstalledGeoIPExtra(
            'Please install the geoip extra to use this feature: pip install camoufox[geoip]'
        )


def download_mmdb() -> None:
    """
    Downloads the MaxMind GeoIP2 database.
    """
    geoip_allowed()

    asset_url = MaxMindDownloader(MMDB_REPO).get_asset()

    with open(MMDB_FILE, 'wb') as f:
        webdl(
            asset_url,
            desc='Downloading GeoIP database',
            buffer=f,
        )


def remove_mmdb() -> None:
    """
    Removes the MaxMind GeoIP2 database.
    """
    if not MMDB_FILE.exists():
        rprint("GeoIP database not found.")
        return

    MMDB_FILE.unlink()
    rprint("GeoIP database removed.")


def get_geolocation(ip: str) -> Geolocation:
    """
    Gets the geolocation for an IP address.
    """
    # Check if the database is downloaded
    if not MMDB_FILE.exists():
        download_mmdb()

    # Validate the IP address
    validate_ip(ip)

    with geoip2.database.Reader(str(MMDB_FILE)) as reader:
        resp = reader.city(ip)
        iso_code = cast(str, resp.registered_country.iso_code).upper()
        location = resp.location

        # Check if any required attributes are missing
        if any(not getattr(location, attr) for attr in ('longitude', 'latitude', 'time_zone')):
            raise UnknownIPLocation(f"Unknown IP location: {ip}")

        # Get a statistically correct locale based on the country code
        locale = SELECTOR.from_region(iso_code)

        return Geolocation(
            locale=locale,
            longitude=cast(float, resp.location.longitude),
            latitude=cast(float, resp.location.latitude),
            timezone=cast(str, resp.location.time_zone),
        )


"""
Gets a random language based on the territory code.
"""


def get_unicode_info() -> ET.Element:
    """
    Fetches supplemental data from the territoryInfo.xml file.
    Source: https://raw.githubusercontent.com/unicode-org/cldr/master/common/supplemental/supplementalData.xml
    """
    with open(LOCAL_DATA / 'territoryInfo.xml', 'rb') as f:
        data = ET.XML(f.read())
    assert data is not None, 'Failed to load territoryInfo.xml'
    return data


def _as_float(element: ET.Element, attr: str) -> float:
    """
    Converts an attribute to a float.
    """
    return float(element.get(attr, 0))


class StatisticalLocaleSelector:
    """
    Selects a random locale based on statistical data.
    Takes either a territory code or a language code, and generates a Locale object.
    """

    __slots__ = (
        "_territory_languages",
        "_language_regions",
        "_known_territories",
        "_known_languages",
    )

    def __init__(self):
        root = get_unicode_info()
        territory_lookup: Dict[str, ProbabilityTable] = {}
        language_lookup: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        known_languages: Set[str] = set()
        known_territories: Set[str] = set()

        for territory in root.findall("territory"):
            iso_code = territory.get("type")
            if not iso_code:
                continue

            known_territories.add(iso_code)

            lang_populations = territory.findall("languagePopulation")
            languages: List[str] = []
            percentages: List[float] = []

            literacy = _as_float(territory, "literacyPercent")
            population = _as_float(territory, "population")

            for lang in lang_populations:
                lang_type = lang.get("type")
                if not lang_type:
                    continue

                known_languages.add(lang_type)

                population_percent = _as_float(lang, "populationPercent")
                if population_percent <= 0:
                    continue

                languages.append(lang_type)
                percentages.append(population_percent)
                language_lookup[lang_type].append(
                    (
                        iso_code,
                        population_percent * literacy / 10_000 * population,
                    )
                )

            if languages:
                territory_lookup[iso_code] = self.normalize_probabilities(
                    languages, percentages
                )

        language_tables: Dict[str, ProbabilityTable] = {}
        for lang_type, entries in language_lookup.items():
            regions = [region for region, _ in entries]
            weights = [weight for _, weight in entries]
            try:
                language_tables[lang_type] = self.normalize_probabilities(
                    regions, weights
                )
            except ValueError:
                continue

        self._territory_languages = territory_lookup
        self._language_regions = language_tables
        self._known_territories = known_territories
        self._known_languages = known_languages

    def _load_territory_data(self, iso_code: str) -> ProbabilityTable:
        """
        Calculates a random language based on the territory code,
        based on the probability that a person speaks the language in the territory.
        """
        if iso_code not in self._known_territories:
            raise UnknownTerritory(f"Unknown territory: {iso_code}")

        try:
            return self._territory_languages[iso_code]
        except KeyError as exc:
            raise ValueError(f"No language data found for region: {iso_code}") from exc

    def _load_language_data(self, language: str) -> ProbabilityTable:
        """
        Calculates a random region for a language
        based on the total speakers of the language in that region.
        """
        if language not in self._known_languages:
            raise UnknownLanguage(f"No region data found for language: {language}")

        try:
            return self._language_regions[language]
        except KeyError as exc:
            raise ValueError(
                f"No valid region data found for language: {language}"
            ) from exc

    def normalize_probabilities(
        self, values: Iterable[str], freq: Iterable[float]
    ) -> ProbabilityTable:
        """
        Normalize probabilities.
        """
        paired = [
            (value, weight)
            for value, weight in zip(values, freq)
            if weight > 0
        ]
        if not paired:
            raise ValueError('No positive probability values provided')

        total = sum(weight for _, weight in paired)
        if total <= 0:
            raise ValueError('Probability weights must sum to a positive value')

        inv_total = 1.0 / total
        cumulative: List[float] = []
        normalized_values: List[str] = []
        running = 0.0

        for value, weight in paired:
            running += weight * inv_total
            normalized_values.append(value)
            cumulative.append(running)

        cumulative[-1] = 1.0  # Guard against floating point drift

        return ProbabilityTable(tuple(normalized_values), tuple(cumulative))

    def from_region(self, region: str) -> Locale:
        """
        Get a random locale based on the territory ISO code.
        Returns as a Locale object.
        """
        distribution = self._load_territory_data(region)
        language = _sample_probability(distribution).replace('_', '-')
        return normalize_locale(f"{language}-{region}")

    def from_language(self, language: str) -> Locale:
        """
        Get a random locale based on the language.
        Returns as a Locale object.
        """
        distribution = self._load_language_data(language)
        region = _sample_probability(distribution)
        return normalize_locale(f"{language}-{region}")


SELECTOR = StatisticalLocaleSelector()
