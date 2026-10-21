"""
Ingredient Function Taxonomy Reference Dataset
Represents verified chemical functional categories documented in CosIng (European Commission Cosmetic Ingredient Database)
and CIR (Cosmetic Ingredient Review).
Categories:
- Humectant: Water-attracting / hydrating agents
- Emollient: Skin softening / barrier lipid agents
- Occlusive: Moisture sealants / transepidermal water loss barriers
- Exfoliating agent: Chemical peeling / keratolytic agents (AHAs, BHAs, PHAs, enzymes)
- Antioxidant: Free-radical scavengers / oxidative stress protectors
- Surfactant / Cleansing agent: Surface active emulsifiers and cleansers
- Soothing / Anti-inflammatory: Redness reduction / barrier calming agents
- Sunscreen active: UV absorbing / blocking physical or chemical filters
"""

import os
import pandas as pd

TAXONOMY_RECORDS = [
    # Humectants
    {"ingredient_name": "glycerin", "functional_category": "Humectant", "description": "Classic trihydroxy alcohol humectant pulling moisture into the stratum corneum"},
    {"ingredient_name": "sodium hyaluronate", "functional_category": "Humectant", "description": "Salt of hyaluronic acid with high water binding capacity"},
    {"ingredient_name": "hyaluronic acid", "functional_category": "Humectant", "description": "Glycosaminoglycan humectant capable of holding up to 1000x its weight in water"},
    {"ingredient_name": "butylene glycol", "functional_category": "Humectant", "description": "Small organic alcohol acting as humectant and solvent"},
    {"ingredient_name": "propylene glycol", "functional_category": "Humectant", "description": "Hydrophilic glycol humectant and penetration enhancer"},
    {"ingredient_name": "panthenol", "functional_category": "Humectant", "description": "Provitamin B5 humectant with wound healing and barrier restoring efficacy"},
    {"ingredient_name": "urea", "functional_category": "Humectant", "description": "Component of Natural Moisturizing Factor (NMF) with water-retaining properties"},
    {"ingredient_name": "sorbitol", "functional_category": "Humectant", "description": "Sugar alcohol humectant providing smooth humectancy"},
    {"ingredient_name": "sodium pca", "functional_category": "Humectant", "description": "Amino acid derivative humectant found natively in human skin NMF"},
    {"ingredient_name": "betaine", "functional_category": "Humectant", "description": "Osmolyte humectant balancing skin cellular hydration"},
    {"ingredient_name": "aloe barbadensis leaf juice", "functional_category": "Humectant", "description": "Polysaccharide-rich botanical extract providing hydration"},
    {"ingredient_name": "polyglutamic acid", "functional_category": "Humectant", "description": "Biopolymer humectant forming high water-retention film on skin"},

    # Emollients
    {"ingredient_name": "squalane", "functional_category": "Emollient", "description": "Hydrogenated biomimetic hydrocarbon lipid that softens skin without clogging pores"},
    {"ingredient_name": "caprylic capric triglyceride", "functional_category": "Emollient", "description": "Fractionated coconut oil ester providing lightweight emollience"},
    {"ingredient_name": "cetyl alcohol", "functional_category": "Emollient", "description": "Fatty alcohol acting as non-drying emollient and emulsion stabilizer"},
    {"ingredient_name": "stearyl alcohol", "functional_category": "Emollient", "description": "Long-chain fatty alcohol providing richness and smoothing texture"},
    {"ingredient_name": "cetearyl alcohol", "functional_category": "Emollient", "description": "Mixture of cetyl and stearyl fatty alcohols conditioning the skin"},
    {"ingredient_name": "isopropyl myristate", "functional_category": "Emollient", "description": "Ester emollient enhancing spreadability and product absorption"},
    {"ingredient_name": "jojoba seed oil", "functional_category": "Emollient", "description": "Liquid wax ester closely resembling natural human sebum lipids"},
    {"ingredient_name": "simmondsia chinensis", "functional_category": "Emollient", "description": "Botanical jojoba seed oil providing biocompatible emollience"},
    {"ingredient_name": "helianthus annuus seed oil", "functional_category": "Emollient", "description": "Sunflower oil high in linoleic acid to nourish the skin barrier"},
    {"ingredient_name": "argan oil", "functional_category": "Emollient", "description": "Nutritive vegetable oil packed with essential fatty acids"},
    {"ingredient_name": "ceramide np", "functional_category": "Emollient", "description": "Essential barrier lipid replenishing intercellular cement"},
    {"ingredient_name": "ceramide ap", "functional_category": "Emollient", "description": "Stratum corneum structural lipid maintaining epidermal integrity"},
    {"ingredient_name": "ceramide eop", "functional_category": "Emollient", "description": "Long-chain ester-linked ceramide critical for skin barrier cohesion"},
    {"ingredient_name": "cholesterol", "functional_category": "Emollient", "description": "Sterol lipid acting in synergy with ceramides to restore skin barrier"},

    # Occlusives
    {"ingredient_name": "petrolatum", "functional_category": "Occlusive", "description": "Purified hydrocarbon mixture reducing transepidermal water loss by up to 98%"},
    {"ingredient_name": "mineral oil", "functional_category": "Occlusive", "description": "Refined cosmetic-grade hydrocarbon forming a continuous physical barrier"},
    {"ingredient_name": "dimethicone", "functional_category": "Occlusive", "description": "Silicone polymer forming a breathable hydrophobic seal over stratum corneum"},
    {"ingredient_name": "butyrospermum parkii", "functional_category": "Occlusive", "description": "Shea butter rich in stearic and oleic fatty acids creating protective coating"},
    {"ingredient_name": "beeswax", "functional_category": "Occlusive", "description": "Natural ester-wax preventing evaporation of cutaneous moisture"},
    {"ingredient_name": "cera alba", "functional_category": "Occlusive", "description": "Purified natural beeswax offering protective water-resistant barrier"},
    {"ingredient_name": "microcrystalline wax", "functional_category": "Occlusive", "description": "Dense hydrocarbon wax sealing moisture inside dry skin areas"},
    {"ingredient_name": "lanolin", "functional_category": "Occlusive", "description": "Complex lipid wax derived from sheep wool providing semi-permeable occlusion"},
    {"ingredient_name": "paraffin", "functional_category": "Occlusive", "description": "Solid hydrocarbon forming occlusive film on epidermal surface"},

    # Exfoliating agents
    {"ingredient_name": "salicylic acid", "functional_category": "Exfoliating agent", "description": "Lipophilic beta hydroxy acid penetrating sebaceous pores and dissolving keratin plugs"},
    {"ingredient_name": "glycolic acid", "functional_category": "Exfoliating agent", "description": "Smallest alpha hydroxy acid penetrating deeply to unglue dead corneocytes"},
    {"ingredient_name": "lactic acid", "functional_category": "Exfoliating agent", "description": "Alpha hydroxy acid that gently exfoliates while providing humectant hydration"},
    {"ingredient_name": "mandelic acid", "functional_category": "Exfoliating agent", "description": "Large molecular weight AHA ideal for gentle exfoliation on sensitive skin"},
    {"ingredient_name": "gluconolactone", "functional_category": "Exfoliating agent", "description": "Polyhydroxy acid (PHA) offering ultra-gentle superficial exfoliation and antioxidant power"},
    {"ingredient_name": "lactobionic acid", "functional_category": "Exfoliating agent", "description": "Bionic acid providing non-irritating exfoliation and water retention"},
    {"ingredient_name": "papain", "functional_category": "Exfoliating agent", "description": "Proteolytic papaya enzyme dissolving dead surface proteins"},
    {"ingredient_name": "bromelain", "functional_category": "Exfoliating agent", "description": "Pineapple proteolytic enzyme breaking down keratin bonds without abrasion"},
    {"ingredient_name": "retinol", "functional_category": "Exfoliating agent", "description": "Vitamin A derivative stimulating epidermal turnover and cellular renewal"},
    {"ingredient_name": "tretinoin", "functional_category": "Exfoliating agent", "description": "All-trans retinoic acid accelerating keratocyte differentiation and desquamation"},
    {"ingredient_name": "adapalene", "functional_category": "Exfoliating agent", "description": "Third-generation selective retinoid regulating cellular turnover and comedolysis"},

    # Antioxidants
    {"ingredient_name": "ascorbic acid", "functional_category": "Antioxidant", "description": "L-ascorbic acid neutralizing reactive oxygen species and aiding collagen synthesis"},
    {"ingredient_name": "tocopherol", "functional_category": "Antioxidant", "description": "Fat-soluble vitamin E defending cell membrane lipids against peroxidation"},
    {"ingredient_name": "tocopheryl acetate", "functional_category": "Antioxidant", "description": "Stabilized vitamin E ester protecting formulation and skin against oxidation"},
    {"ingredient_name": "niacinamide", "functional_category": "Antioxidant", "description": "Vitamin B3 enhancing cellular NAD+ coenzymes and calming inflammation"},
    {"ingredient_name": "ferulic acid", "functional_category": "Antioxidant", "description": "Plant phenolic antioxidant stabilizing vitamins C and E and doubling photoprotection"},
    {"ingredient_name": "resveratrol", "functional_category": "Antioxidant", "description": "Polyphenolic stilbenoid antioxidant activating sirtuins and fighting oxidative stress"},
    {"ingredient_name": "camellia sinensis", "functional_category": "Antioxidant", "description": "Green tea polyphenols rich in EGCG with potent free-radical scavenging"},
    {"ingredient_name": "ubiquinone", "functional_category": "Antioxidant", "description": "Coenzyme Q10 energizing cellular mitochondria and quenching lipid radicals"},
    {"ingredient_name": "glutathione", "functional_category": "Antioxidant", "description": "Master tripeptide antioxidant restoring other oxidized antioxidants in cells"},
    {"ingredient_name": "azelaic acid", "functional_category": "Antioxidant", "description": "Dicarboxylic acid scavenging free radicals while reducing tyrosine activity"},

    # Surfactants / Cleansers
    {"ingredient_name": "sodium laureth sulfate", "functional_category": "Surfactant / Cleanser", "description": "Ethoxylated anionic surfactant producing rich lather and solubilizing oils"},
    {"ingredient_name": "cocamidopropyl betaine", "functional_category": "Surfactant / Cleanser", "description": "Zwitterionic amphoteric surfactant mitigating irritation of harsher detergents"},
    {"ingredient_name": "sodium cocoyl isethionate", "functional_category": "Surfactant / Cleanser", "description": "Gentle baby-foam surfactant derived from coconut fatty acids"},
    {"ingredient_name": "coco-glucoside", "functional_category": "Surfactant / Cleanser", "description": "Non-ionic alkyl glucoside surfactant biodegradable and mild on skin"},
    {"ingredient_name": "decyl glucoside", "functional_category": "Surfactant / Cleanser", "description": "Ultra-mild non-ionic sugar surfactant suitable for sensitive and baby skin"},
    {"ingredient_name": "lauryl glucoside", "functional_category": "Surfactant / Cleanser", "description": "Non-ionic cleansing surfactant with high viscosity and gentle profile"},
    {"ingredient_name": "sodium lauroyl sarcosinate", "functional_category": "Surfactant / Cleanser", "description": "Mild amino acid surfactant offering effective lather with low barrier disruption"},

    # Soothing / Anti-inflammatory
    {"ingredient_name": "centella asiatica", "functional_category": "Soothing", "description": "Gotu kola botanical rich in madecassoside soothing inflammation and wound repair"},
    {"ingredient_name": "allantoin", "functional_category": "Soothing", "description": "Purine derivative moisturizing, desquamating, and soothing irritated skin"},
    {"ingredient_name": "bisabolol", "functional_category": "Soothing", "description": "Active sesquiterpene from chamomile attenuating cutaneous erythema"},
    {"ingredient_name": "colloidal oatmeal", "functional_category": "Soothing", "description": "Beta-glucan rich avenanthramide complex calming itch and eczema flare-ups"},
    {"ingredient_name": "madecassoside", "functional_category": "Soothing", "description": "Isolated triterpenoid saponin suppressing inflammatory cytokines"},
    {"ingredient_name": "glycyrrhiza glabra", "functional_category": "Soothing", "description": "Licorice root extract rich in glabridin reducing redness and pigmentation"},

    # Sunscreen Actives
    {"ingredient_name": "zinc oxide", "functional_category": "Sunscreen active", "description": "Broad-spectrum inorganic physical filter scattering UVA and UVB radiation"},
    {"ingredient_name": "titanium dioxide", "functional_category": "Sunscreen active", "description": "Mineral physical sunscreen filter providing high UVB and short UVA protection"},
    {"ingredient_name": "avobenzone", "functional_category": "Sunscreen active", "description": "Oil-soluble organic filter absorbing full-spectrum UVA rays"},
    {"ingredient_name": "octinoxate", "functional_category": "Sunscreen active", "description": "Organic chemical filter absorbing UVB wavelengths"},
    {"ingredient_name": "octisalate", "functional_category": "Sunscreen active", "description": "Oil-soluble UVB chemical absorber supplementing sunscreen SPF"},
    {"ingredient_name": "homosalate", "functional_category": "Sunscreen active", "description": "Organic UVB filter commonly used in high-SPF formulations"},
    {"ingredient_name": "tinosorb s", "functional_category": "Sunscreen active", "description": "Broad-spectrum photo-stable chemical filter absorbing UVA and UVB"},
    {"ingredient_name": "tinosorb m", "functional_category": "Sunscreen active", "description": "Hybrid organic particulate filter both absorbing and scattering UV rays"}
]


def create_ingredient_taxonomy_dataset() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dest_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "ingredient_taxonomy.csv")
    df = pd.DataFrame(TAXONOMY_RECORDS)
    df.to_csv(dest_path, index=False)
    print(f"Saved verified ingredient taxonomy dataset ({len(df)} records) to: {dest_path}")
    return dest_path


if __name__ == "__main__":
    create_ingredient_taxonomy_dataset()
