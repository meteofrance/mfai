"""
Utility functions to interact with http servers.
"""

import os
import ssl
import urllib.error
import urllib.request

from tqdm import tqdm


def _get_ssl_context() -> ssl.SSLContext:
    """Return an SSL context for HTTPS downloads.

    Builds a context from the environment CA bundle (honouring
    ``SSL_CERT_FILE``/``SSL_CERT_DIR``) when possible, falling back to the
    default context otherwise.

    Peer verification (``VERIFY_PEER``) is always kept enabled, but the
    ``VERIFY_X509_STRICT`` flag is cleared: newer OpenSSL builds (e.g. the one
    bundled with Python 3.13) reject chains that lack an Authority Key
    Identifier extension, whereas curl and older OpenSSL accept them. This
    makes downloads robust across Python >=3.10 without disabling security.
    """
    cafile = os.environ.get("SSL_CERT_FILE")
    capath = os.environ.get("SSL_CERT_DIR")
    try:
        context = ssl.create_default_context(cafile=cafile or None, capath=capath or None)
    except (OSError, ssl.SSLError):
        context = ssl.create_default_context()
    context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    return context


def download_file(url: str, destination: str, backup_url: str | None = None) -> None:
    """
    Downloads a file from url into destination, on failure will try backup_url if provided.

    Args:
        url: Primary URL to download from.
        destination: Local path where the file is written.
        backup_url: Fallback URL used if the primary download fails.
    """

    ssl_context = _get_ssl_context()

    def _attempt_download(download_url: str) -> bool:
        """
        Attempts to download a file, skipping it if it already exists and is up-to-date.

        Args:
            download_url: URL to download from.

        Returns:
            bool: True on success, False on failure.
        """

        with urllib.request.urlopen(download_url, context=ssl_context) as response:
            # Get the total file size from headers, defaulting to 0 if not present
            file_size = int(response.headers.get("Content-Length", 0))

            # Check if file exists and has the same size
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"File already exists and is up-to-date: {destination}")
                    return True  # Indicate success without re-downloading

            block_size = 1024  # 1 Kilobyte

            # Initialize the progress bar with total file size
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
            return True

    try:
        if _attempt_download(url):
            return
    except (urllib.error.HTTPError, urllib.error.URLError):
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except (urllib.error.HTTPError, urllib.error.URLError):
                pass

        # If we reach here, both attempts have failed
        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
