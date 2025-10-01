
<div align="center">
    <img width="150px" src="https://raw.githubusercontent.com/Animex-App/Animex/refs/heads/main/assets/icon.png" alt="Animex Logo"/>
    <h1 style="border-bottom: none;">Animex [Extension Sources]</h1>
    <p><strong>Your new home for Anime and Manga. Indie-made and designed for the community.</strong></p>
</div>

<div align="center">
    <a href="https://github.com/Animex-App/Animex/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Animex-App/Animex?style=for-the-badge&labelColor=1a1d24"></a>
    <a href="https://github.com/Animex-App/Animex-Desktop/releases"><img alt="Latest Release" src="https://img.shields.io/github/v/release/Animex-App/Animex-Desktop?style=for-the-badge&labelColor=1a1d24"></a>
    <a href="https://github.com/Animex-App/Animex/releases"><img alt="Total Downloads" src="https://img.shields.io/github/downloads/Animex-App/Animex/total?style=for-the-badge&labelColor=1a1d24"></a>
</div>

---

### **Project Status: Pre-Release**

**Estimated Release:** Late July (Date TBD)

> **Important Notice:** Animex does not host any of the content available to users. The platform functions by scraping and serving content from third-party sources. For more information, please see our [Extension Sources](https://github.com/Animex-App/Extension-Servers).

---

### **Extension Server Setup**

The Animex platform requires a self-hosted extension server. You can run this on a local machine or a home lab.

**Setup Instructions:**

1.  **Download:** Get the latest release (ZIP file).
2.  **Extract:** Unzip the file in your desired location.
3.  **Add Modules:** Place the `.module` files (from this repository or other sources) into the `{path}/modules/` directory.
4.  **Install Python:** Ensure you have [Python 3](https://www.python.org/downloads/) installed.
5.  **Run:** Execute the following command to start the server:
    ```bash
    ./start.sh
    ```

---

### **Features**

*   **Modular System:** Easily add and manage content sources.
*   **User-Provided Sources:** Flexibility to integrate your preferred sources.
*   **And much more to come!**

---

### **Currently Available Modules**

| Source | Language | Subtitles | Dubbing | Type |
| :--- | :--- | :--- | :--- | :--- |
| 🇺🇸 9Anime (via GogoAnime) | English | ✅ | ✅ | Streaming |
| *More sources coming soon...* | | | | |

> **Warning:** All data, including streaming content, is obtained by scraping external sources and may be unreliable or subject to change.

---

### **Legal Information**

**Disclaimer**

*   Animex is a tool designed to help users discover and organize anime and manga, utilizing APIs from services like MyAnimeList (Jikan) and Kitsu.
*   The developers of Animex do not host any of the content accessible through the platform. All anime and manga are scraped from and served by third-party providers.
*   The owner and developers of Animex are not liable for the misuse of any content and are not responsible for its dissemination.
*   By using this software, you acknowledge that the content may not originate from a legitimate source and agree that the developers are not responsible for the content provided.
*   For any copyright concerns, please direct your inquiries to the respective content sources.

**License**
---
This project is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.html#license-text).
