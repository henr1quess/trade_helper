# Batch 25-28 Weight/Stack Data Attempt

Attempts to source weight and stack_max values for batches 25-28 from NWDB (https://nwdb.info) continue to fail from the sandbox environment. Every direct request to nwdb.info returns HTTP 403 due to the outbound proxy restrictions, blocking access to the authoritative dataset required to fill the fields accurately.

```
$ curl -I https://nwdb.info/db/item/ancientsporest1
HTTP/1.1 403 Forbidden
```

No alternative local sources in the repository contain the required metadata for these items, so updating `subgrupos_batches.json` would require guessing values, which violates the project rules.
