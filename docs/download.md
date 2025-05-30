
### Download Data

This Python script downloads all files from a TalkBank dataset directory (HTML index page)

It uses your authenticated session via a `cookies.txt` file exported from your browser.

---

### Step 1: Log In and Export Cookies

1. Go to [talkbank](https://media.talkbank.org) and log in.
3. Use the browser extension [**Get cookies.txt LOCALLY**](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) to export cookies.
4. Save it as `cookies.txt` in the script directory.

---

### Step 2: Run the Script

```bash
python download_dataset.py
```

This will:
- Use cookies to authenticate.
- Parse the index page recursively.
- Download all files into `data/Voices-AWS/`.

---


## Output Structure

```
datasets/
└── Voices-AWS/
    ├── interview/
    │   ├── videos/
    │   │   ├── 01f.mp4
    │   │   ├── 02m.mp4
    │   ├── interview_dataset.csv
    │   ├── exclusions.csv
    └── reading/
        └── ...
```


