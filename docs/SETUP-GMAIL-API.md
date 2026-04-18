# Gmail API Setup — Trade Analyst Pipeline

## 1. Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click **Select a project** → **New Project**
3. Name: `tradelog-analyst` (or any name)
4. Click **Create**

## 2. Enable the Gmail API

1. In the project dashboard, go to **APIs & Services** → **Library**
2. Search for **Gmail API**
3. Click **Enable**

## 3. Create OAuth2 Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **OAuth client ID**
3. If prompted, configure the **OAuth consent screen** first:
   - User type: **External** (or Internal if using Google Workspace)
   - App name: `tradelog-analyst`
   - User support email: your email
   - Scopes: add `https://www.googleapis.com/auth/gmail.readonly`
   - Test users: add your Gmail address
   - Save and continue through all steps
4. Back in Credentials, create the OAuth client:
   - Application type: **Desktop app**
   - Name: `tradelog-analyst-desktop`
   - Click **Create**
5. Click **Download JSON** on the created credential
6. Save the file as: `finance/apps/analyst/_credentials/client_secret.json`

## 4. First-Time Authentication

Run the analyst pipeline once to trigger the OAuth flow:

```bash
uv run python -m finance.apps analyst --dry-run
```

This will:
1. Open your browser for Google sign-in
2. Ask you to grant read-only Gmail access
3. Save the refresh token to `finance/apps/analyst/_credentials/gmail_token.json`

Subsequent runs use the saved token automatically — no browser needed.

## 5. Gmail Filter Setup

Create a Gmail filter to label market-related emails:

1. In Gmail, click the search bar → **Show search options**
2. Set **From** to your market email senders, separated by OR:
   ```
   alerts@barchart.com OR newsletter@briefing.com OR ...
   ```
3. Click **Create filter**
4. Check: **Apply the label** → create label `market`
5. Check: **Skip the Inbox (Archive it)**
6. Click **Create filter**

The pipeline will fetch all emails with the `market` label received since the last run.

## 6. File Locations

| File | Purpose | Committed? |
|------|---------|------------|
| `_credentials/client_secret.json` | Google OAuth2 client secret | Yes (private repo) |
| `_credentials/gmail_token.json` | Refresh token (created on first run) | Yes (private repo) |

## 7. Scopes

The pipeline requests **read-only** access:
- `https://www.googleapis.com/auth/gmail.readonly`

It cannot send, delete, or modify emails. The `market` label restriction is enforced in code.

## 8. Troubleshooting

**"Access blocked: This app's request is invalid"**
→ Make sure your email is added as a test user in the OAuth consent screen.

**"Token has been expired or revoked"**
→ Delete `_credentials/gmail_token.json` and re-run to re-authenticate.

**"Gmail API has not been used in project..."**
→ Enable the Gmail API in Google Cloud Console (step 2).
