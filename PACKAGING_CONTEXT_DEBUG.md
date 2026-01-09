# Testing Product Context for Packaging Analysis

## Problem

Product context shows "Product context not available" even though data exists in OpenFacts database.

## Testing Steps

### Step 1: Verify OpenFacts Data Structure

First, check what data is actually available from OpenFacts:

```powershell
cd ai-service
python test_openfacts_structure.py
```

This will:

- Fetch product data directly from OpenFacts API
- Show all relevant fields (brands, product_name, categories, packagings)
- Save the full response to `openfacts_response.json` for inspection
- Identify any missing fields

**What to check:**

- Is `brands` field present and not empty?
- Is `product_name` or `name` field present?
- Is `categories` field present?
- Does `packagings` array exist and have items?
- Does any packaging item have `food_contact: 1`?

---

### Step 2: Test Full Packaging Flow

Test the complete packaging analysis flow with detailed logging:

```powershell
# Make sure your AI service is running
python main.py

# In another terminal:
python test_packaging_context.py
```

This will:

1. Call `/lookup-barcode` to get product data
2. Call `/barcode/packaging/separate` to extract materials and context
3. Call `/barcode/packaging/describe` with the extracted context
4. Show exactly what data is being passed at each step

**Check the server logs** for these log lines:

- `[PACKAGING-SEPARATE] Product context extracted:` - Shows what was extracted
- `[PACKAGING-DESCRIBE] Received product context:` - Shows what was received
- `[PACKAGING-ANALYZE] Building context from parameters:` - Shows what's being used
- `[PACKAGING-ANALYZE] Final product context:` - Shows the final context string
- `[PACKAGING-ANALYZE] Full prompt being sent to OpenAI:` - Shows the complete prompt

---

## Common Issues & Fixes

### Issue 1: Fields are empty strings instead of missing

**Symptom:** Logs show `Brand: ''` instead of `Brand: 'Cadbury'`

**Cause:** OpenFacts returns empty strings `""` for some fields instead of omitting them

**Fix:** Update the extraction logic to treat empty strings as missing:

```python
# In /barcode/packaging/separate route
brand_name = product_data.get("brands", "") or None  # Treat empty string as None
product_name = product_data.get("product_name", "") or product_data.get("name", "") or None
categories = product_data.get("categories", "") or None
```

### Issue 2: Frontend not passing context to /describe

**Symptom:** `/separate` logs show context but `/describe` logs show all None/empty

**Cause:** Frontend needs to be updated to pass `product_context` from `/separate` response to `/describe` request

**Fix:** Update frontend code to include context fields in the describe request

### Issue 3: Packagings array structure different

**Symptom:** `food_contact` is always None even though packagings exist

**Possible causes:**

- Field might be `food_contact: "yes"` instead of `food_contact: 1`
- Field might be in a different location

**Debug:** Check `openfacts_response.json` to see actual structure

**Fix if needed:**

```python
# Try different ways to detect food contact
if pkg.get("food_contact") in [1, "1", "yes", True]:
    food_contact = True
```

---

## How to Use Different Barcodes for Testing

Edit either test file and change:

```python
TEST_BARCODE = "YOUR_BARCODE_HERE"
```

Some test barcodes:

- `5000159461122` - Cadbury Dairy Milk (chocolate bar)
- `737628064502` - Coca-Cola (beverage)
- `0078000113464` - Cheerios (cereal)

---

## Expected Output

**When working correctly, you should see:**

```
[PACKAGING-ANALYZE] Final product context:
Brand: Cadbury
Product: Dairy Milk
Category: Candy chocolate bars
Food Contact: YES - direct food contact
```

**In the AI prompt:**

```
PRODUCT CONTEXT:
Brand: Cadbury
Product: Dairy Milk
Category: Candy chocolate bars
Food Contact: YES - direct food contact
```

**NOT this:**

```
Product context not available
```

---

## Debugging Checklist

- [ ] Ran `test_openfacts_structure.py` and verified data exists
- [ ] Checked `openfacts_response.json` for actual field structure
- [ ] Ran `test_packaging_context.py` with server logs visible
- [ ] Verified context is extracted in `/separate` route
- [ ] Verified context is received in `/describe` route
- [ ] Verified context is built correctly in `analyze_packaging_material`
- [ ] Checked the final prompt sent to OpenAI
- [ ] Verified frontend is passing context from `/separate` to `/describe`
