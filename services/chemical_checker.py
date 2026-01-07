"""
Chemical Red-Flag Detection System
Based on Hippiekit's comprehensive toxin database
"""

from typing import List, Dict, Any, Set
import re


# Severity levels
CRITICAL = "critical"  # -20 points
HIGH = "high"          # -10 points
MODERATE = "moderate"  # -5 points
LOW = "low"            # -2 points


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
    severity_order = {CRITICAL: 0, HIGH: 1, MODERATE: 2, LOW: 3}
    result = sorted(
        found.values(),
        key=lambda x: (severity_order[x["severity"]], x["chemical"])
    )
    
    return result


def calculate_safety_score(flagged_chemicals: List[Dict[str, Any]]) -> int:
    """
    Calculate safety score (0-100) based on flagged chemicals.
    Base score 100, subtract: -20 per critical, -10 per high, -5 per moderate, -2 per low.
    
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
    for chem_name, chem_data in CHEMICALS.items():
        category = chem_data["category"]
        severity = chem_data["severity"]
        
        # Only include critical and high severity for condensed list
        if severity in [CRITICAL, HIGH]:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(chem_name)
    
    # Build condensed text
    lines = ["CHEMICAL RED FLAGS (flag if found):"]
    for category, chemicals in sorted(by_category.items()):
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
