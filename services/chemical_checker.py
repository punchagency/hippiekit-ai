"""
Chemical Red-Flag Detection System
Based on Hippiekit's comprehensive toxin database
"""

from typing import List, Dict, Any, Set
import re
import os
from pathlib import Path


# Severity levels
CRITICAL = "critical"      # -20 points
HIGH = "high"              # -10 points
MODERATE = "moderate"      # -5 points
QUESTIONABLE = "questionable"  # -3 points (not harmful, but not "clean" - synthetic additives)
LOW = "low"                # -2 points


# ========== SUSTAINABLE PACKAGING DEFINITIONS ==========
# Materials that are eco-friendly and acceptable for Hippiekit recommendations

SUSTAINABLE_PACKAGING_MATERIALS = {
    # Glass - infinitely recyclable, no chemical leaching
    "glass",
    "glass bottle",
    "glass jar",
    "amber glass",
    "recycled glass",
    
    # Metal - highly recyclable
    "aluminum",
    "aluminium",
    "aluminum can",
    "steel",
    "stainless steel",
    "tin",
    "metal",
    "metal can",
    "metal tin",
    
    # Paper & Cardboard - biodegradable
    "paper",
    "cardboard",
    "paperboard",
    "kraft paper",
    "recycled paper",
    "recycled cardboard",
    "paper bag",
    "paper wrap",
    "paper carton",
    "cardboard box",
    
    # Compostable/Biodegradable materials
    "compostable",
    "biodegradable",
    "plant-based packaging",
    "bamboo",
    "wood",
    "wooden",
    "cork",
    "hemp",
    "cotton",
    "jute",
    "burlap",
    "beeswax wrap",
    
    # Sustainable alternatives
    "refillable",
    "reusable",
    "zero waste",
    "package-free",
    "bulk",
    "naked packaging",
}

# Materials that are NOT sustainable and should be avoided
NON_SUSTAINABLE_PACKAGING_MATERIALS = {
    # Plastics
    "plastic",
    "plastic bottle",
    "plastic bag",
    "plastic wrap",
    "plastic container",
    "plastic pouch",
    "plastic tub",
    "shrink wrap",
    "cling film",
    "cling wrap",
    "cellophane",  # Most commercial cellophane is plastic
    
    # Specific plastic types
    "pet",
    "pete",
    "hdpe",
    "pvc",
    "ldpe",
    "pp",
    "polypropylene",
    "polystyrene",
    "styrofoam",
    "eps",  # Expanded polystyrene
    "bpa",
    "bps",
    "polycarbonate",
    "vinyl",
    
    # Plastic numbers
    "plastic #1",
    "plastic #2",
    "plastic #3",
    "plastic #4",
    "plastic #5",
    "plastic #6",
    "plastic #7",
    
    # Mixed materials (hard to recycle)
    "tetra pak",  # While better than plastic, still problematic
    "foil-lined",
    "multi-layer",
    "laminated",
    
    # Single-use items
    "single-use",
    "disposable plastic",
    "blister pack",
}

# Keywords that indicate sustainable/eco-friendly products
SUSTAINABILITY_POSITIVE_KEYWORDS = {
    "organic",
    "usda organic",
    "certified organic",
    "non-gmo",
    "non gmo",
    "eco-friendly",
    "eco friendly",
    "sustainable",
    "sustainably sourced",
    "fair trade",
    "fairtrade",
    "plastic-free",
    "plastic free",
    "zero waste",
    "refillable",
    "reusable",
    "biodegradable",
    "compostable",
    "recyclable",
    "recycled",
    "glass bottle",
    "glass jar",
    "aluminum",
    "bpa-free",
    "bpa free",
    "phthalate-free",
    "phthalate free",
    "natural",
    "plant-based",
    "vegan",
    "cruelty-free",
    "cruelty free",
    "b corp",
    "b-corp",
    "ewg verified",
    "made safe",
    "clean label",
    "green certified",
    "carbon neutral",
    "climate neutral",
}

# Keywords that indicate unsustainable products (should avoid recommending)
SUSTAINABILITY_NEGATIVE_KEYWORDS = {
    "plastic bottle",
    "plastic container",
    "plastic packaging",
    "single-use",
    "disposable",
    "conventional",
    "non-recyclable",
    "styrofoam",
    "pvc",
    "vinyl",
}


def is_packaging_sustainable(packaging_text: str) -> dict:
    """
    Check if packaging description indicates sustainable materials.
    
    Args:
        packaging_text: Description of packaging materials
        
    Returns:
        {
            "is_sustainable": bool,
            "sustainable_materials": list of detected sustainable materials,
            "non_sustainable_materials": list of detected non-sustainable materials,
            "sustainability_score": int (0-100),
            "recommendation": str
        }
    """
    if not packaging_text:
        return {
            "is_sustainable": False,  # Unknown = assume not sustainable
            "sustainable_materials": [],
            "non_sustainable_materials": [],
            "sustainability_score": 50,
            "recommendation": "Packaging information not available"
        }
    
    text_lower = packaging_text.lower()
    
    sustainable_found = []
    non_sustainable_found = []
    
    # Check for sustainable materials
    for material in SUSTAINABLE_PACKAGING_MATERIALS:
        if material in text_lower:
            sustainable_found.append(material)
    
    # Check for non-sustainable materials
    for material in NON_SUSTAINABLE_PACKAGING_MATERIALS:
        if material in text_lower:
            non_sustainable_found.append(material)
    
    # Calculate sustainability score
    score = 50  # Start neutral
    score += len(sustainable_found) * 15  # Boost for each sustainable material
    score -= len(non_sustainable_found) * 25  # Penalty for non-sustainable
    score = max(0, min(100, score))  # Clamp to 0-100
    
    # Determine overall sustainability
    is_sustainable = len(sustainable_found) > 0 and len(non_sustainable_found) == 0
    
    # Generate recommendation
    if is_sustainable:
        recommendation = "Eco-friendly packaging using sustainable materials"
    elif len(non_sustainable_found) > 0:
        recommendation = f"Contains non-sustainable packaging: {', '.join(non_sustainable_found[:3])}"
    else:
        recommendation = "Packaging sustainability uncertain"
    
    return {
        "is_sustainable": is_sustainable,
        "sustainable_materials": list(set(sustainable_found)),
        "non_sustainable_materials": list(set(non_sustainable_found)),
        "sustainability_score": score,
        "recommendation": recommendation
    }


def check_product_sustainability(description: str, name: str = "") -> dict:
    """
    Check if a product description indicates it's sustainable/eco-friendly.
    
    Args:
        description: Product description text
        name: Product name (optional, for additional context)
        
    Returns:
        {
            "is_likely_sustainable": bool,
            "positive_indicators": list of found sustainability keywords,
            "negative_indicators": list of found problematic keywords,
            "sustainability_score": int (0-100)
        }
    """
    combined_text = f"{name} {description}".lower()
    
    positive_found = []
    negative_found = []
    
    for keyword in SUSTAINABILITY_POSITIVE_KEYWORDS:
        if keyword in combined_text:
            positive_found.append(keyword)
    
    for keyword in SUSTAINABILITY_NEGATIVE_KEYWORDS:
        if keyword in combined_text:
            negative_found.append(keyword)
    
    # Calculate score
    score = 50
    score += len(positive_found) * 10
    score -= len(negative_found) * 20
    score = max(0, min(100, score))
    
    return {
        "is_likely_sustainable": len(positive_found) > len(negative_found) and len(negative_found) == 0,
        "positive_indicators": list(set(positive_found)),
        "negative_indicators": list(set(negative_found)),
        "sustainability_score": score
    }


def get_sustainability_requirements_prompt() -> str:
    """
    Get the sustainability requirements text to inject into AI prompts.
    This ensures AI recommendations align with Hippiekit's sustainable values.
    """
    return """
HIPPIEKIT SUSTAINABILITY REQUIREMENTS (CRITICAL - MUST FOLLOW):

Hippiekit is a ZERO PLASTIC platform. ALL product recommendations MUST be 100% plastic-free.

⚠️ IMPORTANT: "Glass bottle" is NOT automatically plastic-free! 
Many glass bottles have PLASTIC CAPS, PLASTIC SEALS, or PLASTIC SHRINK WRAP.

✅ TRULY PLASTIC-FREE PACKAGING (ONLY recommend these):
- Glass with METAL/CORK/WOOD closures (NOT plastic caps)
- Aluminum cans with aluminum pull-tabs (NOT plastic pour spouts)
- Metal tins with metal lids
- Paper/cardboard boxes with paper tape
- Products with CORK stoppers (wine-style closures)
- Bamboo containers with bamboo lids
- Ceramic with cork/wood lids
- Refillable containers at zero-waste stores
- Package-free/bulk options (bring your own container)
- Beeswax wraps
- Compostable plant-based materials (verified, not "bio-plastic")

❌ NOT ACCEPTABLE - DO NOT RECOMMEND:
- Glass bottles with PLASTIC CAPS (very common - AVOID)
- Glass jars with PLASTIC LINED LIDS
- Metal cans with PLASTIC POUR SPOUTS
- "Recyclable plastic" - still plastic, AVOID
- Paper products with PLASTIC LINING (coffee cups, milk cartons)
- Tetra Pak (has plastic layer)
- Products with PLASTIC SHRINK WRAP or SEALS
- Products with PLASTIC LABELS
- ANY product where plastic touches the food/product
- "BPA-free plastic" - still plastic, AVOID
- Bioplastics/PLA - still plastic-derived, AVOID unless certified compostable

🏆 TRULY ZERO-PLASTIC BRANDS TO PRIORITIZE:
- Meliora (cleaning products in metal tins)
- Ethique (solid bars in cardboard)
- Plaine Products (aluminum bottles with metal pumps)
- Package Free Shop brands
- Zero Waste Store brands
- Brands explicitly marketed as "100% plastic-free"
- Local farmers market vendors
- Bulk bin stores (bring your own container)

FOR OLIVE OIL SPECIFICALLY:
✅ ACCEPT: Glass bottle with CORK or METAL POUR SPOUT
✅ ACCEPT: Metal tin (common in European imports)
❌ REJECT: Glass bottle with plastic cap (most commercial olive oils)
❌ REJECT: Glass bottle with plastic pour spout insert

FOR FOOD PRODUCTS:
✅ ACCEPT: Glass jars with metal twist-off lids (check lid doesn't have plastic liner)
✅ ACCEPT: Metal cans (check no plastic pour spout)
✅ ACCEPT: Paper/cardboard boxes with no plastic window
❌ REJECT: Any plastic wrapper, cap, seal, liner, or component

FOR PERSONAL CARE:
✅ ACCEPT: Solid bars in cardboard/paper (shampoo bars, soap bars)
✅ ACCEPT: Metal tins (deodorant, balms)
✅ ACCEPT: Glass with metal pump (rare but exists - Plaine Products)
❌ REJECT: Plastic tubes, plastic pump bottles, plastic caps

IF NO PLASTIC-FREE OPTION EXISTS:
1. First choice: Recommend a DIY/homemade alternative with recipe
2. Second choice: Recommend buying from a zero-waste bulk store
3. Third choice: Recommend the user contact brands to request plastic-free packaging
4. Fourth choice: Recommend buying in largest size to minimize packaging per unit

NEVER recommend a product with ANY plastic component, even if it's "mostly" sustainable.
"""


# Chemical Database with Categories, Aliases, and Severity
CHEMICALS = {
    # ========== PRESERVATIVES & FORMALDEHYDE-RELEASERS ==========
    "Methylparaben": {"category": "Preservatives", "severity": HIGH, "aliases": ["methyl paraben", "methyl-paraben"]},
    "Propylparaben": {"category": "Preservatives", "severity": HIGH, "aliases": ["propyl paraben", "propyl-paraben"]},
    "Butylparaben": {"category": "Preservatives", "severity": HIGH, "aliases": ["butyl paraben", "butyl-paraben"]},
    "Ethylparaben": {"category": "Preservatives", "severity": HIGH, "aliases": ["ethyl paraben", "ethyl-paraben"]},
    "DMDM Hydantoin": {"category": "Preservatives", "severity": CRITICAL, "aliases": ["dmdm"]},
    "Diazolidinyl Urea": {"category": "Preservatives", "severity": CRITICAL, "aliases": ["diazolidinyl"]},
    "Imidazolidinyl Urea": {"category": "Preservatives", "severity": CRITICAL, "aliases": ["imidazolidinyl"]},
    "Quaternium-15": {"category": "Preservatives", "severity": CRITICAL, "aliases": ["quaternium 15"]},
    "Formaldehyde": {"category": "Preservatives", "severity": CRITICAL, "aliases": []},
    "Propyl Gallate": {"category": "Preservatives", "severity": HIGH, "aliases": []},
    "TBHQ": {"category": "Preservatives", "severity": HIGH, "aliases": ["tertiary butylhydroquinone", "tert-butylhydroquinone"]},
    "BHA": {"category": "Preservatives", "severity": HIGH, "aliases": ["butylated hydroxyanisole"]},
    "BHT": {"category": "Preservatives", "severity": HIGH, "aliases": ["butylated hydroxytoluene"]},
    "Sodium Benzoate": {"category": "Preservatives", "severity": MODERATE, "aliases": []},
    "Potassium Benzoate": {"category": "Preservatives", "severity": MODERATE, "aliases": []},
    "Potassium Sorbate": {"category": "Preservatives", "severity": MODERATE, "aliases": []},
    "Phenoxyethanol": {"category": "Preservatives", "severity": MODERATE, "aliases": []},
    "Methylisothiazolinone": {"category": "Preservatives", "severity": HIGH, "aliases": ["mi", "mit"]},
    "Methylchloroisothiazolinone": {"category": "Preservatives", "severity": HIGH, "aliases": ["mci"]},
    
    # ========== SURFACTANTS & FOAMING AGENTS ==========
    "Sodium Lauryl Sulfate": {"category": "Surfactants", "severity": HIGH, "aliases": ["sls", "sodium dodecyl sulfate"]},
    "Sodium Laureth Sulfate": {"category": "Surfactants", "severity": HIGH, "aliases": ["sles", "sodium lauryl ether sulfate"]},
    "Ammonium Lauryl Sulfate": {"category": "Surfactants", "severity": HIGH, "aliases": ["als"]},
    "Cocamidopropyl Betaine": {"category": "Surfactants", "severity": MODERATE, "aliases": ["capb"]},
    "Polysorbate 20": {"category": "Surfactants", "severity": MODERATE, "aliases": ["tween 20"]},
    "Polysorbate 60": {"category": "Surfactants", "severity": MODERATE, "aliases": ["tween 60"]},
    "Polysorbate 80": {"category": "Surfactants", "severity": MODERATE, "aliases": ["tween 80"]},
    "Nonylphenol Ethoxylates": {"category": "Surfactants", "severity": CRITICAL, "aliases": ["npe"]},
    "Octoxynol-9": {"category": "Surfactants", "severity": HIGH, "aliases": []},
    
    # ========== FRAGRANCE / PARFUM ==========
    "Fragrance": {"category": "Fragrance", "severity": HIGH, "aliases": ["parfum", "perfume"]},
    "Linalool": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Limonene": {"category": "Fragrance", "severity": MODERATE, "aliases": ["d-limonene"]},
    "Geraniol": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Citronellol": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Benzyl Acetate": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Benzyl Alcohol": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Coumarin": {"category": "Fragrance", "severity": HIGH, "aliases": []},
    "Eugenol": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Methyl Eugenol": {"category": "Fragrance", "severity": HIGH, "aliases": []},
    "Acetaldehyde": {"category": "Fragrance", "severity": HIGH, "aliases": []},
    "Ethyl Vanillin": {"category": "Fragrance", "severity": LOW, "aliases": []},
    "Isoamyl Acetate": {"category": "Fragrance", "severity": MODERATE, "aliases": []},
    "Galaxolide": {"category": "Fragrance", "severity": HIGH, "aliases": ["hhcb", "synthetic musk"]},
    "Tonalide": {"category": "Fragrance", "severity": HIGH, "aliases": ["ahtn"]},
    "Musk Ketone": {"category": "Fragrance", "severity": CRITICAL, "aliases": ["nitromusk"]},
    
    # ========== SYNTHETIC DYES & COLORANTS ==========
    "Red 40": {"category": "Dyes", "severity": HIGH, "aliases": ["allura red", "e129"]},
    "Yellow 5": {"category": "Dyes", "severity": HIGH, "aliases": ["tartrazine", "e102"]},
    "Yellow 6": {"category": "Dyes", "severity": HIGH, "aliases": ["sunset yellow", "e110"]},
    "Blue 1": {"category": "Dyes", "severity": MODERATE, "aliases": ["brilliant blue", "e133", "blue no. 1", "blue no 1", "fd&c blue 1"]},
    "Blue 2": {"category": "Dyes", "severity": MODERATE, "aliases": ["indigotine", "e132"]},
    "Green 3": {"category": "Dyes", "severity": HIGH, "aliases": []},
    "Red 3": {"category": "Dyes", "severity": HIGH, "aliases": ["erythrosine", "e127"]},
    "Red 40": {"category": "Dyes", "severity": HIGH, "aliases": ["allura red", "e129", "red no. 40", "red no 40", "fd&c red 40"]},
    "Yellow 5": {"category": "Dyes", "severity": HIGH, "aliases": ["tartrazine", "e102", "yellow no. 5", "yellow no 5", "fd&c yellow 5"]},
    "Yellow 6": {"category": "Dyes", "severity": HIGH, "aliases": ["sunset yellow", "e110", "yellow no. 6", "yellow no 6", "fd&c yellow 6"]},
    "Caramel Color": {"category": "Dyes", "severity": MODERATE, "aliases": ["e150", "caramel"]},
    "Coal Tar Dyes": {"category": "Dyes", "severity": CRITICAL, "aliases": ["coal tar"]},
    "Carbon Black": {"category": "Dyes", "severity": CRITICAL, "aliases": []},
    "Titanium Dioxide": {"category": "Dyes", "severity": MODERATE, "aliases": ["ci 77891", "nano titanium dioxide"]},
    
    # ========== ARTIFICIAL FLAVORS ==========
    "Diacetyl": {"category": "Flavors", "severity": HIGH, "aliases": []},
    "Acetoin": {"category": "Flavors", "severity": MODERATE, "aliases": []},
    "Vanillin": {"category": "Flavors", "severity": LOW, "aliases": []},
    "Benzaldehyde": {"category": "Flavors", "severity": MODERATE, "aliases": []},
    "Butyl Acetate": {"category": "Flavors", "severity": MODERATE, "aliases": []},
    "Methyl Anthranilate": {"category": "Flavors", "severity": MODERATE, "aliases": []},
    "Allyl Hexanoate": {"category": "Flavors", "severity": MODERATE, "aliases": []},
    "Ethyl Butyrate": {"category": "Flavors", "severity": MODERATE, "aliases": []},
    "Propylene Glycol": {"category": "Flavors", "severity": MODERATE, "aliases": ["pg", "1,2-propanediol"]},
    "Triacetin": {"category": "Flavors", "severity": LOW, "aliases": ["glyceryl triacetate"]},
    "Flavourings": {"category": "Flavors", "severity": HIGH, "aliases": ["flavorings", "flavoring", "flavouring", "natural flavourings", "natural flavorings", "artificial flavourings", "artificial flavorings"]},
    "Natural Flavor": {"category": "Flavors", "severity": MODERATE, "aliases": ["natural flavors", "natural flavour", "natural flavours"]},
    "Artificial Flavor": {"category": "Flavors", "severity": HIGH, "aliases": ["artificial flavors", "artificial flavour", "artificial flavours"]},
    
    # ========== ULTRA-PROCESSED ADDITIVES & STARCHES ==========
    "Maltodextrin": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Dextrin": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Modified Starch": {"category": "Processing Aids", "severity": HIGH, "aliases": ["modified corn starch", "modified food starch", "modified tapioca starch", "modified potato starch"]},
    "Modified Corn Starch": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Modified Food Starch": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Corn Syrup Solids": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "High Fructose Corn Syrup": {"category": "Processing Aids", "severity": CRITICAL, "aliases": ["hfcs"]},
    "Glucose-Fructose Syrup": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Invert Sugar": {"category": "Processing Aids", "severity": MODERATE, "aliases": ["invert sugar syrup"]},
    "Hydrolyzed Vegetable Protein": {"category": "Processing Aids", "severity": MODERATE, "aliases": ["hvp", "hydrolyzed soy protein"]},
    "Yeast Extract": {"category": "Processing Aids", "severity": MODERATE, "aliases": []},
    "Monosodium Glutamate": {"category": "Processing Aids", "severity": HIGH, "aliases": ["msg", "e621"]},
    "Disodium Guanylate": {"category": "Processing Aids", "severity": MODERATE, "aliases": ["e627"]},
    "Disodium Inosinate": {"category": "Processing Aids", "severity": MODERATE, "aliases": ["e631"]},
    
    # ========== EMULSIFIERS & STABILIZERS ==========
    "Carrageenan": {"category": "Thickeners", "severity": MODERATE, "aliases": ["e407"]},
    "Xanthan Gum": {"category": "Thickeners", "severity": LOW, "aliases": ["e415"]},
    "Guar Gum": {"category": "Thickeners", "severity": LOW, "aliases": ["e412"]},
    "Locust Bean Gum": {"category": "Thickeners", "severity": LOW, "aliases": ["e410", "carob gum"]},
    "Cellulose Gum": {"category": "Thickeners", "severity": MODERATE, "aliases": ["cmc", "e466", "carboxymethyl cellulose"]},
    "Microcrystalline Cellulose": {"category": "Thickeners", "severity": MODERATE, "aliases": ["e460"]},
    "Mono and Diglycerides": {"category": "Emulsifiers", "severity": MODERATE, "aliases": ["e471", "monoglycerides", "diglycerides"]},
    "Soy Lecithin": {"category": "Emulsifiers", "severity": MODERATE, "aliases": ["lecithin", "e322"]},
    "Polysorbate 65": {"category": "Emulsifiers", "severity": MODERATE, "aliases": []},
    "Sodium Stearoyl Lactylate": {"category": "Emulsifiers", "severity": MODERATE, "aliases": ["ssl", "e481"]},
    "DATEM": {"category": "Emulsifiers", "severity": MODERATE, "aliases": ["e472e", "diacetyl tartaric acid esters"]},
    
    # ========== SWEETENERS ==========
    "Aspartame": {"category": "Sweeteners", "severity": CRITICAL, "aliases": []},
    "Sucralose": {"category": "Sweeteners", "severity": HIGH, "aliases": ["splenda"]},
    "Saccharin": {"category": "Sweeteners", "severity": HIGH, "aliases": []},
    "Acesulfame K": {"category": "Sweeteners", "severity": HIGH, "aliases": ["acesulfame potassium", "ace-k"]},
    "Neotame": {"category": "Sweeteners", "severity": HIGH, "aliases": []},
    "Advantame": {"category": "Sweeteners", "severity": HIGH, "aliases": []},
    "Cyclamate": {"category": "Sweeteners", "severity": CRITICAL, "aliases": []},
    "Xylitol": {"category": "Sweeteners", "severity": LOW, "aliases": []},
    "Erythritol": {"category": "Sweeteners", "severity": LOW, "aliases": []},
    "Sorbitol": {"category": "Sweeteners", "severity": LOW, "aliases": []},
    "Maltitol": {"category": "Sweeteners", "severity": LOW, "aliases": []},
    "Mannitol": {"category": "Sweeteners", "severity": LOW, "aliases": []},
    
    # ========== SEED OILS ==========
    "Canola Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": ["rapeseed oil"]},
    "Soybean Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": ["soy oil"]},
    "Corn Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": []},
    "Cottonseed Oil": {"category": "Seed Oils", "severity": HIGH, "aliases": []},
    "Sunflower Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": []},
    "Safflower Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": []},
    "Grapeseed Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": []},
    "Rice Bran Oil": {"category": "Seed Oils", "severity": MODERATE, "aliases": []},
    
    # ========== TEXTILE TOXINS ==========
    "Polyester": {"category": "Textiles", "severity": HIGH, "aliases": ["pet fabric"]},
    "Terephthalic Acid": {"category": "Textiles", "severity": HIGH, "aliases": []},
    "Ethylene Glycol": {"category": "Textiles", "severity": HIGH, "aliases": []},
    "Antimony Trioxide": {"category": "Textiles", "severity": CRITICAL, "aliases": []},
    "Nylon": {"category": "Textiles", "severity": HIGH, "aliases": ["polyamide"]},
    "Caprolactam": {"category": "Textiles", "severity": HIGH, "aliases": []},
    "Adipic Acid": {"category": "Textiles", "severity": MODERATE, "aliases": []},
    "Rayon": {"category": "Textiles", "severity": HIGH, "aliases": ["viscose"]},
    "Carbon Disulfide": {"category": "Textiles", "severity": CRITICAL, "aliases": []},
    "Acrylic": {"category": "Textiles", "severity": HIGH, "aliases": []},
    "Acrylonitrile": {"category": "Textiles", "severity": CRITICAL, "aliases": []},
    "Spandex": {"category": "Textiles", "severity": MODERATE, "aliases": ["elastane", "lycra"]},
    "Toluene Diisocyanate": {"category": "Textiles", "severity": CRITICAL, "aliases": ["tdi"]},
    
    # ========== PLASTIC PACKAGING & ADDITIVES ==========
    "PET": {"category": "Plastics", "severity": MODERATE, "aliases": ["polyethylene terephthalate", "plastic #1"]},
    "PVC": {"category": "Plastics", "severity": CRITICAL, "aliases": ["polyvinyl chloride", "vinyl", "plastic #3"]},
    "Polycarbonate": {"category": "Plastics", "severity": HIGH, "aliases": ["pc", "plastic #7"]},
    "BPA": {"category": "Plastics", "severity": CRITICAL, "aliases": ["bisphenol a"]},
    "BPS": {"category": "Plastics", "severity": CRITICAL, "aliases": ["bisphenol s"]},
    "BPF": {"category": "Plastics", "severity": CRITICAL, "aliases": ["bisphenol f"]},
    "Phthalates": {"category": "Plastics", "severity": CRITICAL, "aliases": ["dehp", "dbp", "dinp"]},
    "Styrene": {"category": "Plastics", "severity": HIGH, "aliases": []},
    "Vinyl Chloride": {"category": "Plastics", "severity": CRITICAL, "aliases": []},
    "PLA": {"category": "Plastics", "severity": MODERATE, "aliases": ["polylactic acid", "plant plastic"]},
    "1,4-Dioxane": {"category": "Plastics", "severity": CRITICAL, "aliases": ["dioxane"]},
    "Triphenyl Phosphate": {"category": "Plastics", "severity": HIGH, "aliases": ["tphp"]},
    
    # ========== SUNSCREEN & UV FILTERS ==========
    "Oxybenzone": {"category": "Sunscreen", "severity": CRITICAL, "aliases": ["bp-3", "benzophenone-3"]},
    "Octinoxate": {"category": "Sunscreen", "severity": HIGH, "aliases": ["omc", "ethylhexyl methoxycinnamate"]},
    "Homosalate": {"category": "Sunscreen", "severity": HIGH, "aliases": []},
    "Octocrylene": {"category": "Sunscreen", "severity": HIGH, "aliases": []},
    "Avobenzone": {"category": "Sunscreen", "severity": MODERATE, "aliases": ["butyl methoxydibenzoylmethane"]},
    "Retinyl Palmitate": {"category": "Sunscreen", "severity": HIGH, "aliases": ["vitamin a palmitate"]},
    "Octisalate": {"category": "Sunscreen", "severity": MODERATE, "aliases": ["ethylhexyl salicylate"]},
    "PABA": {"category": "Sunscreen", "severity": HIGH, "aliases": ["para-aminobenzoic acid"]},
    
    # ========== PFAS ==========
    "PTFE": {"category": "PFAS", "severity": CRITICAL, "aliases": ["teflon", "polytetrafluoroethylene"]},
    "PFOA": {"category": "PFAS", "severity": CRITICAL, "aliases": ["perfluorooctanoic acid"]},
    "PFOS": {"category": "PFAS", "severity": CRITICAL, "aliases": ["perfluorooctane sulfonate", "scotchgard"]},
    
    # ========== TOOTHPASTE-SPECIFIC ==========
    "Triclosan": {"category": "Antimicrobials", "severity": CRITICAL, "aliases": []},
    "Fluoride": {"category": "Toothpaste", "severity": MODERATE, "aliases": ["sodium fluoride", "stannous fluoride"]},
    "Carrageenan": {"category": "Thickeners", "severity": MODERATE, "aliases": []},
    
    # ========== COSMETIC-SPECIFIC ==========
    "Dimethicone": {"category": "Silicones", "severity": MODERATE, "aliases": []},
    "Talc": {"category": "Cosmetics", "severity": HIGH, "aliases": []},
    "Petrolatum": {"category": "Cosmetics", "severity": MODERATE, "aliases": ["petroleum jelly"]},
    "Mineral Oil": {"category": "Cosmetics", "severity": MODERATE, "aliases": []},
    "DEA": {"category": "Cosmetics", "severity": HIGH, "aliases": ["diethanolamine"]},
    "TEA": {"category": "Cosmetics", "severity": HIGH, "aliases": ["triethanolamine"]},
    "MEA": {"category": "Cosmetics", "severity": HIGH, "aliases": ["monoethanolamine"]},
    "Resorcinol": {"category": "Cosmetics", "severity": HIGH, "aliases": []},
    "Toluene": {"category": "Cosmetics", "severity": CRITICAL, "aliases": []},
    "Ammonia": {"category": "Cosmetics", "severity": HIGH, "aliases": []},
    
    # ========== PESTICIDE RESIDUES ==========
    "Glyphosate": {"category": "Pesticides", "severity": CRITICAL, "aliases": ["roundup"]},
    "Atrazine": {"category": "Pesticides", "severity": CRITICAL, "aliases": []},
    "Chlorpyrifos": {"category": "Pesticides", "severity": CRITICAL, "aliases": []},
    "Malathion": {"category": "Pesticides", "severity": HIGH, "aliases": []},
    "Paraquat": {"category": "Pesticides", "severity": CRITICAL, "aliases": []},
    "Imidacloprid": {"category": "Pesticides", "severity": HIGH, "aliases": ["neonicotinoid"]},
    "DDT": {"category": "Pesticides", "severity": CRITICAL, "aliases": []},
    
    # ========== HEAVY METALS ==========
    "Lead": {"category": "Heavy Metals", "severity": CRITICAL, "aliases": ["pb"]},
    "Mercury": {"category": "Heavy Metals", "severity": CRITICAL, "aliases": ["hg"]},
    "Arsenic": {"category": "Heavy Metals", "severity": CRITICAL, "aliases": ["as"]},
    "Cadmium": {"category": "Heavy Metals", "severity": CRITICAL, "aliases": ["cd"]},
    "Chromium": {"category": "Heavy Metals", "severity": HIGH, "aliases": ["cr", "hexavalent chromium"]},
    "Aluminum": {"category": "Heavy Metals", "severity": MODERATE, "aliases": ["aluminium"]},
    
    # ========== FOOD PROCESSING AIDS ==========
    "Azodicarbonamide": {"category": "Processing Aids", "severity": HIGH, "aliases": ["ada", "yoga mat chemical"]},
    "Potassium Bromate": {"category": "Processing Aids", "severity": CRITICAL, "aliases": []},
    "Sodium Nitrite": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Sodium Nitrate": {"category": "Processing Aids", "severity": HIGH, "aliases": []},
    "Benzoyl Peroxide": {"category": "Processing Aids", "severity": MODERATE, "aliases": []},
    
    # ========== SYNTHETIC MINERALS & ADDITIVES (QUESTIONABLE) ==========
    # These are FDA-approved but not naturally sourced - commonly found in processed waters
    "Calcium Chloride": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["cacl2", "e509"]},
    "Magnesium Chloride": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["mgcl2", "e511"]},
    "Potassium Bicarbonate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["khco3", "e501"]},
    "Sodium Selenate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["na2seo4"]},
    "Sodium Bicarbonate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["baking soda", "e500"]},
    "Potassium Chloride": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["kcl", "e508"]},
    "Magnesium Sulfate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["epsom salt", "mgso4", "e518"]},
    "Calcium Sulfate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["gypsum", "caso4", "e516"]},
    "Sodium Phosphate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["trisodium phosphate", "e339"]},
    "Potassium Phosphate": {"category": "Synthetic Minerals", "severity": QUESTIONABLE, "aliases": ["e340"]},
}


def build_search_patterns() -> Dict[str, Dict]:
    """
    Build regex patterns for efficient chemical matching.
    Returns dict mapping lowercase patterns to chemical info.
    """
    patterns = {}
    
    for chem_name, chem_data in CHEMICALS.items():
        # Add main name
        pattern_key = chem_name.lower()
        patterns[pattern_key] = {
            "name": chem_name,
            **chem_data
        }
        
        # Add aliases
        for alias in chem_data["aliases"]:
            alias_key = alias.lower()
            patterns[alias_key] = {
                "name": chem_name,  # Store original name
                **chem_data
            }
    
    return patterns


# Build search patterns once at module load
SEARCH_PATTERNS = build_search_patterns()

# Pre-compile regex patterns for better performance
COMPILED_PATTERNS = {}
for pattern in SEARCH_PATTERNS.keys():
    COMPILED_PATTERNS[pattern] = {
        "standard": re.compile(r'\b' + re.escape(pattern) + r's?\b'),
        "parentheses": re.compile(r'\(' + re.escape(pattern) + r's?\)'),
        "hyphen_slash": re.compile(r'[-/]' + re.escape(pattern) + r's?\b')
    }


def check_ingredients(ingredient_text: str) -> List[Dict[str, Any]]:
    """
    Check ingredient text for harmful chemicals.
    Enhanced to catch E-numbers in parentheses like 'Red 40 (E129)' or 'Tartrazine (E102)'.
    
    Args:
        ingredient_text: Text containing ingredients (from label, database, etc.)
    
    Returns:
        List of detected chemicals with their info
    """
    if not ingredient_text:
        return []
    
    # Normalize text
    text_lower = ingredient_text.lower()
    
    # Track found chemicals (use set to avoid duplicates)
    found: Dict[str, Dict] = {}
    
    # Search for each pattern
    for pattern, chem_info in SEARCH_PATTERNS.items():
        # Use pre-compiled regex patterns for better performance
        compiled = COMPILED_PATTERNS[pattern]
        
        # Check all patterns using pre-compiled regex
        if (compiled["standard"].search(text_lower) or 
            compiled["parentheses"].search(text_lower) or 
            compiled["hyphen_slash"].search(text_lower)):
            
            chem_name = chem_info["name"]
            
            # Only add once even if multiple aliases match
            if chem_name not in found:
                found[chem_name] = {
                    "chemical": chem_name,
                    "category": chem_info["category"],
                    "severity": chem_info["severity"],
                    "matched_as": pattern
                }
    
    # Convert to list and sort by severity
    severity_order = {CRITICAL: 0, HIGH: 1, MODERATE: 2, QUESTIONABLE: 3, LOW: 4}
    result = sorted(
        found.values(),
        key=lambda x: (severity_order.get(x["severity"], 5), x["chemical"])
    )
    
    return result


def calculate_safety_score(flagged_chemicals: List[Dict[str, Any]]) -> int:
    """
    Calculate safety score (0-100) based on flagged chemicals.
    Base score 100, subtract: -20 per critical, -10 per high, -5 per moderate, -3 per questionable, -2 per low.
    
    Args:
        flagged_chemicals: List of dicts from check_ingredients()
    
    Returns:
        Safety score (0-100)
    """
    score = 100
    
    for chem in flagged_chemicals:
        severity = chem["severity"]
        
        if severity == CRITICAL:
            score -= 20
        elif severity == HIGH:
            score -= 10
        elif severity == MODERATE:
            score -= 5
        elif severity == QUESTIONABLE:
            score -= 3
        elif severity == LOW:
            score -= 2
    
    # Clamp to 0-100
    return max(0, min(100, score))


def generate_recommendations(
    flagged_chemicals: List[Dict[str, Any]],
    product_category: str = None
) -> Dict[str, List[str]]:
    """
    Generate actionable recommendations based on flagged chemicals.
    
    Args:
        flagged_chemicals: List of detected harmful chemicals
        product_category: Optional product category for specific advice
    
    Returns:
        Dict with 'avoid', 'look_for', and 'certifications' lists
    """
    avoid = set()
    look_for = set()
    certifications = set()
    
    # Group chemicals by category
    categories_found = set(chem["category"] for chem in flagged_chemicals)
    
    # Category-specific recommendations
    if "Preservatives" in categories_found:
        avoid.add("Products with parabens, formaldehyde-releasers, or DMDM Hydantoin")
        look_for.add("Preserved with natural alternatives like rosemary extract or vitamin E")
    
    if "Surfactants" in categories_found:
        avoid.add("Sodium Lauryl Sulfate (SLS) and Sodium Laureth Sulfate (SLES)")
        look_for.add("'Sulfate-free' or 'SLS-free' labels")
        look_for.add("Gentle cleansers with coconut-derived surfactants")
    
    if "Fragrance" in categories_found:
        avoid.add("Products listing 'Fragrance' or 'Parfum' (hides 50-100 chemicals)")
        look_for.add("'Fragrance-free' or scented only with essential oils")
        look_for.add("Products that list all fragrance components")
    
    if "Dyes" in categories_found:
        avoid.add("Synthetic dyes (Red 40, Yellow 5, Yellow 6)")
        look_for.add("Plant-based colorants like beet juice, turmeric, or spirulina")
    
    if "Sweeteners" in categories_found:
        avoid.add("Artificial sweeteners (aspartame, sucralose, saccharin)")
        look_for.add("Natural sweeteners like honey, maple syrup, or stevia")
    
    if "Seed Oils" in categories_found:
        avoid.add("Highly processed seed oils (canola, soybean, corn oil)")
        look_for.add("Cold-pressed olive oil, coconut oil, or avocado oil")
    
    if "Plastics" in categories_found or "PFAS" in categories_found:
        avoid.add("Plastic packaging (especially #3 PVC and #7 polycarbonate)")
        avoid.add("Non-stick coatings and PFAS (Teflon, Scotchgard)")
        look_for.add("Glass, stainless steel, or PFAS-free certified packaging")
    
    if "Sunscreen" in categories_found:
        avoid.add("Chemical UV filters (oxybenzone, octinoxate)")
        look_for.add("Mineral sunscreens with zinc oxide or titanium dioxide (non-nano)")
    
    if "Pesticides" in categories_found:
        avoid.add("Non-organic produce and conventionally grown ingredients")
        certifications.add("USDA Organic")
        certifications.add("Non-GMO Project Verified")
    
    if "Heavy Metals" in categories_found:
        avoid.add("Products without third-party testing for heavy metals")
        look_for.add("Brands that publish heavy metal test results")
    
    if "Synthetic Minerals" in categories_found:
        avoid.add("Products with synthetic mineral additives used for pH buffering or fortification")
        look_for.add("Naturally mineralized spring water or filtered water")
        look_for.add("Products with minerals from natural sources")
    
    # General certifications
    if len(flagged_chemicals) >= 3:
        certifications.add("EWG Verified")
        certifications.add("Made Safe Certified")
        certifications.add("Leaping Bunny (Cruelty-Free)")
        certifications.add("B Corp Certified")
    
    # Product-specific advice
    if product_category:
        cat_lower = product_category.lower()
        
        if "shampoo" in cat_lower or "soap" in cat_lower or "body wash" in cat_lower:
            look_for.add("Castile soap-based or coconut oil-based formulas")
        
        if "lotion" in cat_lower or "cream" in cat_lower:
            look_for.add("Plant-based oils (jojoba, shea butter, coconut)")
        
        if "food" in cat_lower or "snack" in cat_lower:
            look_for.add("Short ingredient lists with recognizable foods")
            certifications.add("Whole Foods Premium Body Care standards")
    
    return {
        "avoid": sorted(list(avoid)),
        "look_for": sorted(list(look_for)),
        "certifications": sorted(list(certifications))
    }


def get_condensed_chemical_list() -> str:
    """
    Get condensed chemical list for Vision API prompt (grouped by category).
    Limited to most critical chemicals to fit token limits.
    """
    # Group by category
    by_category = {}
    questionable_category = {}
    
    for chem_name, chem_data in CHEMICALS.items():
        category = chem_data["category"]
        severity = chem_data["severity"]
        
        # Separate questionable into its own group
        if severity == QUESTIONABLE:
            if category not in questionable_category:
                questionable_category[category] = []
            questionable_category[category].append(chem_name)
        # Only include critical and high severity for harmful condensed list
        elif severity in [CRITICAL, HIGH]:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(chem_name)
    
    # Build condensed text
    lines = ["CHEMICAL RED FLAGS (flag if found):"]
    for category, chemicals in sorted(by_category.items()):
        lines.append(f"\n{category}: {', '.join(sorted(chemicals))}")
    
    # Add questionable chemicals section
    if questionable_category:
        lines.append("\n\nQUESTIONABLE ADDITIVES (approved but not natural):")
        for category, chemicals in sorted(questionable_category.items()):
            lines.append(f"\n{category}: {', '.join(sorted(chemicals))}")
    
    return "\n".join(lines)


# Example usage for testing
if __name__ == "__main__":
    # Test with sample ingredient list
    test_ingredients = """
    Water, Sodium Lauryl Sulfate, Cocamidopropyl Betaine, Fragrance,
    Methylparaben, Propylparaben, Red 40, Yellow 5, BPA
    """
    
    flags = check_ingredients(test_ingredients)
    score = calculate_safety_score(flags)
    recs = generate_recommendations(flags, "shampoo")
    
    print("=== DETECTED CHEMICALS ===")
    for flag in flags:
        print(f"  {flag['chemical']} ({flag['category']}) - Severity: {flag['severity']}")
    
    print(f"\n=== SAFETY SCORE: {score}/100 ===")
    
    print("\n=== RECOMMENDATIONS ===")
    print("\n❌ AVOID:")
    for item in recs["avoid"]:
        print(f"  - {item}")
    
    print("\n✅ LOOK FOR:")
    for item in recs["look_for"]:
        print(f"  - {item}")
    
    print("\n🏆 TRUSTED CERTIFICATIONS:")
    for item in recs["certifications"]:
        print(f"  - {item}")


# ========== LOAD COMPREHENSIVE CHEMICAL DATABASE FROM FILE ==========

# Cache for the loaded chemical database
_HARMFUL_CHEMICALS_TEXT_CACHE = None

def load_harmful_chemicals_db() -> str:
    """
    Load the comprehensive harmful chemicals list from list_of_chemicals.txt.
    This is cached in memory after first load for performance.
    
    Returns:
        Full text content of the chemical database for AI prompt injection
    """
    global _HARMFUL_CHEMICALS_TEXT_CACHE
    
    # Return cached version if already loaded
    if _HARMFUL_CHEMICALS_TEXT_CACHE is not None:
        return _HARMFUL_CHEMICALS_TEXT_CACHE
    
    try:
        # Get path to list_of_chemicals.txt (same directory as this file's parent)
        current_dir = Path(__file__).parent.parent
        chemicals_file = current_dir / "list_of_chemicals.txt"
        
        if not chemicals_file.exists():
            # Fallback: try relative path
            chemicals_file = Path("list_of_chemicals.txt")
        
        if not chemicals_file.exists():
            raise FileNotFoundError(f"Could not find list_of_chemicals.txt at {chemicals_file}")
        
        # Load and cache the file content
        with open(chemicals_file, 'r', encoding='utf-8') as f:
            _HARMFUL_CHEMICALS_TEXT_CACHE = f.read()
        
        print(f"[CHEMICAL-DB] Loaded {len(_HARMFUL_CHEMICALS_TEXT_CACHE)} characters from {chemicals_file}")
        return _HARMFUL_CHEMICALS_TEXT_CACHE
        
    except Exception as e:
        print(f"[CHEMICAL-DB] Error loading harmful chemicals database: {e}")
        # Return empty string as fallback
        return ""
