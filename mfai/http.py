"""
Utility functions to interact with http servers.
"""

import os
import ssl
import urllib.error
import urllib.request

from tqdm import tqdm


def _get_ssl_context(*, reduced: bool = False) -> ssl.SSLContext:
    """Return an SSL context for HTTPS downloads.

    Builds a context from the environment CA bundle (honouring
    ``SSL_CERT_FILE``/``SSL_CERT_DIR``) when possible, falling back to the
    default context otherwise.

    Peer verification (``VERIFY_PEER``) is always kept enabled. When
    ``reduced`` is True, the ``VERIFY_X509_STRICT`` flag is additionally
    cleared, since newer OpenSSL builds (e.g. the one bundled with Python
    3.13) reject chains that lack an Authority Key Identifier extension,
    whereas curl and older OpenSSL accept them.

    Args:
        reduced: Whether to relax the strict certificate chain check.

    Returns:
        ssl.SSLContext: Configured context for HTTPS downloads.
    """
    cafile = os.environ.get("SSL_CERT_FILE")
    capath = os.environ.get("SSL_CERT_DIR")
    try:
        context = ssl.create_default_context(
            cafile=cafile or None, capath=capath or None
        )
    except (OSError, ssl.SSLError):
        context = ssl.create_default_context()
    if reduced:
        context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def _attempt_download(
    download_url: str, destination: str, ssl_context: ssl.SSLContext
) -> None:
    """Download a file, skipping it if it already exists and is up-to-date.

    Args:
        download_url: URL to download from.
        destination: Local path where the file is written.
        ssl_context: SSL context used for the HTTPS connection.

    Raises:
        urllib.error.HTTPError: If the server responds with an HTTP error.
        urllib.error.URLError: If the URL cannot be opened.
    """
    with urllib.request.urlopen(download_url, context=ssl_context) as response:
        file_size = int(response.headers.get("Content-Length", 0))

        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        block_size = 1024  # 1 Kilobyte

        progress_bar_description = os.path.basename(download_url)
        with tqdm(
            total=file_size,
            unit="iB",
            unit_scale=True,
            desc=progress_bar_description,
        ) as progress_bar:
            with open(destination, "wb") as file:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))


def download_file(url: str, destination: str, backup_url: str | None = None) -> None:
    """
    Downloads a file from url into destination, on failure will try
    backup_url if provided, then reduce ssl security requirements.

    Args:
        url: Primary URL to download from.
        destination: Local path where the file is written.
        backup_url: Fallback URL used if the primary download fails.
    """
    attempts: list[tuple[str, ssl.SSLContext]] = [(url, _get_ssl_context())]
    if backup_url is not None:
        attempts.append((backup_url, _get_ssl_context()))
    reduced_context = _get_ssl_context(reduced=True)
    attempts.append((url, reduced_context))
    if backup_url is not None:
        attempts.append((backup_url, reduced_context))

    for download_url, ssl_context in attempts:
        try:
            _attempt_download(download_url, destination, ssl_context)
            return
        except (urllib.error.HTTPError, urllib.error.URLError):
            print(f"Failed to download from {download_url}")
            continue

    error_message = (
        f"Failed to download from both primary URL ({url})"
        f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
        "\nCheck your internet connection or the file availability.\n"
        "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
    )
    print(error_message)
