# Third-Party Notices

## jiji262/douyin-downloader

- Source: https://github.com/jiji262/douyin-downloader
- Baseline commit: `ad7338fdc474c14e2063370540a362cd8a953b43`
- License: MIT
- Integration: pinned source under `upstream/jiji`, accessed through an application-owned gateway.
- Compatibility: the gateway expects upstream package version `2.0.0` at the pinned commit.
- Credential boundary: application cookies stay in Windows Credential Manager or request memory;
  the upstream file-backed `CookieManager` is not used to persist application credentials.

## JoeanAmier/TikTokDownloader

- Source: https://github.com/JoeanAmier/TikTokDownloader
- Reference commit: `c02c165c26d165fe409e2312f444a80fff3af868`
- License: GPL-3.0
- Integration: optional separately installed HTTP service only; its source is not bundled or imported.
