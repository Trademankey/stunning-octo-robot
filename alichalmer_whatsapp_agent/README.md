# AliChalmer Coffee Shop Sales Bot

This service is a business-specific WhatsApp sales bot for `AliChalmer.com`. Its sole purpose is to help small coffee shops and similar businesses in the United States and Mexico carry Ali Chalmer Coffee.

## What this MVP includes

- FastAPI webhook for WhatsApp Cloud API verification and inbound message handling
- English and Spanish onboarding with language persistence
- Main menu for roasts, wholesale, samples, pricing, shipping, and sales handoff
- Business-specific intent routing with short sales-first responses
- Lead capture for coffee shop wholesale and human sales escalation
- Database layer that uses SQLite locally and `DATABASE_URL` in production
- Knowledge base markdown files for brand, roasts, shipping, wholesale policy, and escalation rules
- `/preview` endpoint for testing flows without connecting Meta

## Project layout

```text
alichalmer_whatsapp_agent/
├── app/
│   ├── agent.py
│   ├── config.py
│   ├── intents.py
│   ├── knowledge_base.py
│   ├── language.py
│   ├── main.py
│   ├── store.py
│   ├── types.py
│   └── services/
├── knowledge/
├── tests/
├── .env.example
└── requirements.txt
```

## Quick start

```bash
cd alichalmer_whatsapp_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Local conversation preview:

```bash
curl -X POST http://127.0.0.1:8080/preview \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"demo-1","message":"Hola, tengo una cafeteria y quiero muestras"}'
```

## WhatsApp setup

1. Create or use your Meta Business Manager and WhatsApp Business Account.
2. Create a Meta app with WhatsApp enabled.
3. Add your webhook URL to the app and use `WHATSAPP_VERIFY_TOKEN` for verification.
4. Set `WHATSAPP_ACCESS_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` in `.env`.
5. Point the webhook to `/webhook`.
6. Create approved message templates in Meta for outbound updates such as sample follow-up and wholesale quote reminders.

## WooCommerce setup

WooCommerce settings are still available in the codebase for future store integrations, but the current bot does not perform retail order lookup. Keep the bot focused on wholesale coffee shop qualification unless the sales workflow intentionally expands.

## Lead capture options

By default, wholesale leads and sales handoff data are stored in local SQLite at `data/agent_state.db`.

If `DATABASE_URL` is set, the service uses that instead. This is the recommended production path for Render Postgres, Railway Postgres, Neon, Supabase, or another managed PostgreSQL database.

If you want Google Sheets or a CRM right away, point `LEAD_WEBHOOK_URL` to:

- a Google Apps Script web app
- Zapier or Make webhook
- your CRM intake endpoint

Each capture sends structured JSON with the lead kind, language, customer ID, and collected fields. The primary lead kind is `coffee_shop_wholesale`.

## Deployment files

This repo now includes:

- `Dockerfile` for container deploys
- `../render.yaml` for Render Blueprint deploys from the repo root
- `railway.toml` for Railway config-as-code

## Deploy on Render

1. Push the repo to GitHub.
2. In Render, create a Blueprint using `render.yaml` from the repo root.
3. Render will create:
   - a web service for the WhatsApp agent
   - a Postgres database
4. Fill in the required secret env vars in the Render dashboard:
   - `WHATSAPP_VERIFY_TOKEN`
   - `WHATSAPP_ACCESS_TOKEN`
   - `WHATSAPP_PHONE_NUMBER_ID`
   - `LEAD_WEBHOOK_URL` if you want leads pushed to a CRM, Google Sheet, Zapier, or Make
5. After deploy, use the public URL plus `/webhook` in Meta.

Render notes:

- The service binds on `0.0.0.0` and reads `PORT`.
- `DATABASE_URL` is injected from the managed Postgres instance.
- The blueprint points Render at the service Dockerfile inside `alichalmer_whatsapp_agent/`.

## Deploy on Railway

1. Push the repo to GitHub.
2. Create a new Railway service from the repo.
3. Set the Root Directory to `/alichalmer_whatsapp_agent`.
4. In Railway config-as-code, point to `/alichalmer_whatsapp_agent/railway.toml`.
5. Add a PostgreSQL plugin or database service and expose its connection string as `DATABASE_URL`.
6. Add the same WhatsApp and lead-capture secrets listed above.
7. Deploy and confirm `/health` responds.

Railway notes:

- The included `railway.toml` starts Uvicorn directly.
- If you prefer Docker on Railway, the included `Dockerfile` is ready.
- Keep the service public so Meta can reach the webhook.

## Conversation design notes

- The six-item menu is sent as a WhatsApp interactive list because button messages support fewer quick replies than the sales menu needs.
- Every path should lead toward roasts, samples, wholesale pricing, U.S./Mexico shipping, or a sales handoff.
- The bot never invents prices, discounts, minimum order quantities, shipping rates, or delivery promises.
- The wholesale flow captures name, business, location, business type, number of locations, monthly volume, roast interest, email, and phone.
- Retail orders, returns, books, legal language, and direct person requests trigger human sales handoff instead of a support flow.

## Suggested next upgrades

- Push coffee shop leads directly into HubSpot, Zoho, Google Sheets, or a sales inbox
- Add outbound template sends for sample follow-up and wholesale quote reminders
- Add a real wholesale price sheet once pricing and minimums are approved
- Add sales analytics on sample requests, quote requests, handoff rate, and conversion rate
