# Error Analysis

## Ministral-3:3b

### Performance

- ACCURACY: 0.6666666666666666
- PRECISION: 0.9
- RECALL: 0.5625
- FALSE POSITIVES (non-matches classified as exact matches): 1
- FALSE NEGATIVES (exact matches classified as non-matches): 7

### False Positives (non-matches classified as exact matches)

#### FP 1

- Observation: llm first classifies in correctly as exact_match, then realizes its own error too late in explanation.
- Fix: Put explanation before classification.

```json
{'example_id': 1163641,
 'product_info': {'product_brand': 'KODAK',
                  'product_bullet_point': 'Basic matte paper for arts, craft '
                                          'and snapshots\n'
                                          'Instant dry: No smearing or '
                                          'smudging\n'
                                          'Guaranteed to work with any inkjet '
                                          'printer\n'
                                          'When a picture on a screen just '
                                          'isn’t enough, trust Kodak, the #1 '
                                          'brand of photo paper',
                  'product_color': 'White',
                  'product_description': None,
                  'product_id': 'B000EZ0CTK',
                  'product_title': 'Kodak Photo Paper for inkjet printers, '
                                   'Matte Finish, 7 mil thickness, 100 sheets, '
                                   '8.5” x 11” (8318164)'},
 'query_correction': None,
 'query_info': {'query': 'kodak photo paper 8.5 x 11 glossy',
                'query_id': 58953},
 'query_product_match': {'match_classification': <MatchClassification.EXACT_MATCH: 'exact_match'>,
                         'reasoning': 'The product details explicitly match '
                                      'all the specifications in the query: '
                                      "'kodak photo paper 8.5 x 11 glossy'. "
                                      'While the product description mentions '
                                      "'matte finish' instead of 'glossy,' the "
                                      "product title and dimensions ('8.5” x "
                                      "11”') align perfectly with the query. "
                                      'However, since the query explicitly '
                                      "asks for 'glossy' and the product is "
                                      "'matte,' there is a discrepancy. Upon "
                                      're-evaluation, the product does not '
                                      'match the glossy finish specified in '
                                      'the query. Thus, the correct '
                                      'classification should be **not an exact '
                                      'match** due to the finish type '
                                      'mismatch.'}}
```
                                      
### FALSE NEGATIVES (exact matches classified as non-matches)

#### FN 1

- Observation: llm interprets "exact match" to require not going beyond the query, in addition to not contradicting the query.
- Fix: clarify language around "exact match" - queries are an exact match when their specs are a subset or proper subset of the product specs

```json
{'example_id': 142653,
 'product_info': {'product_brand': 'Allmax Battery',
                  'product_bullet_point': '★ Maximum Power – Allmax Maximum '
                                          'Power AA Batteries are premium '
                                          'grade, high-performance alkaline '
                                          'batteries built with EnergyCircle '
                                          'Technology to supply maximum power '
                                          'to your electronic devices.\n'
                                          '★ Ultra Long-Lasting – Allmax AA '
                                          'batteries are considered “ultra '
                                          'long-lasting” and have been '
                                          'independently tested to outperform '
                                          'standard alkaline batteries, which '
                                          'means you can also expect these '
                                          'batteries to last longer.\n'
                                          '★ 10-Year Shelf Life – Allmax AA '
                                          'batteries are always fresh and '
                                          'maintain power for up to 10 years '
                                          'in proper storage, which makes '
                                          'Allmax the perfect choice to buy in '
                                          'large quantity for long-term '
                                          'storage and emergency preparedness '
                                          'kits.\n'
                                          '★ Leakproof Design – Allmax AA '
                                          'batteries are built from the ground '
                                          'up to seal in materials and protect '
                                          'your electronic devices, which '
                                          'makes Allmax the perfect choice to '
                                          'power all your devices around the '
                                          'house, the office, or outdoors for '
                                          'mobile power.\n'
                                          '★ Device Compatible – Allmax AA '
                                          'batteries power devices such as '
                                          'remote controls, flashlights, toys, '
                                          'Xbox controllers, digital cameras, '
                                          'wireless mice, wireless keyboards, '
                                          'radios, bubble machines, door '
                                          'locks, clocks, electric pencil '
                                          'sharpeners, toothbrushes, laser '
                                          'levels, calculators, garage door '
                                          'openers, carbon monoxide detectors, '
                                          'alarm sensors, blood pressure '
                                          'monitors, meters, scanners, baby '
                                          'monitors, screwdrivers, blood '
                                          'glucose monitors, face scrubbers, '
                                          'and more!',
                  'product_color': None,
                  'product_description': None,
                  'product_id': 'B07F7RH8D4',
                  'product_title': 'Allmax AA Maximum Power Alkaline Batteries '
                                   '(100 Count Bulk Pack) – Ultra Long-Lasting '
                                   'Double A Battery, 10-Year Shelf Life, '
                                   'Leak-Proof, Device Compatible – Powered by '
                                   'EnergyCircle Technology (1.5V)'},
 'query_correction': None,
 'query_info': {'query': 'aa batteries 100 pack', 'query_id': 6014},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies 'AA batteries 100 "
                                      "pack,' which strictly refers to a pack "
                                      'containing exactly 100 AA batteries. '
                                      'While the product details mention a '
                                      "'100 Count Bulk Pack,' the additional "
                                      "specifications such as 'Maximum Power,' "
                                      "'Ultra Long-Lasting,' '10-Year Shelf "
                                      "Life,' 'Leakproof Design,' and 'Device "
                                      "Compatible' are not explicitly required "
                                      'by the query. Thus, the product does '
                                      'not exactly match the query '
                                      'specifications.'}}

```

#### FN 2

- Observation: llm notes correctly that though the title says 100 count, the bullet points say 50 count, therefore categorizes as (correctly/incorrectly?) as non-match.
- Fix: Ambiguous. Likely the product is a 2 x 50 pack, but not explicitly stated. Prioritize title over description? Or leave as is?

```
{'example_id': 142661,
 'product_info': {'product_brand': 'Energizer',
                  'product_bullet_point': 'Made in USA\n'
                                          'Up to 10 years shelf life\n'
                                          'Zero Mercury\n'
                                          'Energizer MAX AA batteries 50 '
                                          'count\n'
                                          'Bulk Packaging',
                  'product_color': None,
                  'product_description': 'Energizer is the battery you count '
                                         "on every day to power your family's "
                                         'toys, home electronics and '
                                         'flashlights is even better. '
                                         'Energizer MAX AA batteries now hold '
                                         'their power for 10 years. Energizer '
                                         'proprietary Power Seal Technology is '
                                         "assurance that you'll have power you "
                                         "can rely on, whether you're "
                                         "capturing baby's first steps, "
                                         'competing in the final level of a '
                                         'video game marathon, or preparing '
                                         'for a storm on the horizon. A 10 '
                                         'year shelf life means that you can '
                                         "be confident you'll always have "
                                         "power when you need it most. that's "
                                         'positivenergy! Features and '
                                         "Functions: World's 1st Zero Mercury "
                                         'AA Alkaline Battery: Energizer is a '
                                         'leader in the industry in powering '
                                         "people's lives responsibly Up to 10 "
                                         'years shelf life Made in USA',
                  'product_id': 'B00KMDL8U6',
                  'product_title': 'Energizer AA Max Alkaline E91 Batteries '
                                   'Made in USA - Expiration 12/2024 or Later '
                                   '- 100 Count'},
 'query_correction': None,
 'query_info': {'query': 'aa batteries 100 pack', 'query_id': 6014},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies 'aa batteries 100 "
                                      "pack,' but the product description "
                                      "lists 'Energizer MAX AA batteries 50 "
                                      "count' in the bullet points, not 100. "
                                      "Although the product mentions 'Bulk "
                                      "Packaging,' the explicit count of 100 "
                                      'is not confirmed or stated clearly in '
                                      'the product details. Thus, the product '
                                      'does not exactly match the query '
                                      'specification of a 100-pack.'}}
```

#### FN 3

- Observation: llm is overthinking it, quibbling over whether the batteries are packaged in a 100-pack or just 100 units
- Fix: this feels like weakness in the model's common sense more than anything

```
{'example_id': 142663,
 'product_info': {'product_brand': 'Powermax',
                  'product_bullet_point': '100 AA BATTERIES: Includes 100 AA '
                                          'super alkaline bulk batteries with '
                                          '1.5 volts of long lasting power\n'
                                          'EVERYDAY DEVICES: ACDelco AA '
                                          'batteries deliver optimal power to '
                                          'common every day household devices '
                                          'such as clocks, remote controls, '
                                          'game controllers, digital cameras, '
                                          'computer mice, toys and more\n'
                                          'LONG LASTING POWER: ACDelco super '
                                          'alkaline batteries deliver long '
                                          'lasting maximum power that is '
                                          'consistent and reliable\n'
                                          '10-YEAR SHELF LIFE: All ACDelco '
                                          'alkaline batteries are built to '
                                          'last with an impressive ten-year '
                                          'shelf life\n'
                                          'ENVIRONMENTAL IMPACT: Our 1.5-volt '
                                          'alkline batteries contain no '
                                          'mercury or cadmium, making them a '
                                          'smart choice for your family and '
                                          'for the environment',
                  'product_color': 'Blue',
                  'product_description': None,
                  'product_id': 'B004SCA15K',
                  'product_title': 'ACDelco 100-Count AA Batteries, Maximum '
                                   'Power Super Alkaline Battery, 10-Year '
                                   'Shelf Life, Recloseable Packaging'},
 'query_correction': None,
 'query_info': {'query': 'aa batteries 100 pack', 'query_id': 6014},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies 'AA batteries 100 "
                                      "pack,' which implies a pack containing "
                                      'exactly 100 AA batteries. While the '
                                      "product description mentions '100 AA "
                                      "super alkaline bulk batteries,' the "
                                      "phrasing 'bulk batteries' could imply "
                                      'individual units rather than a '
                                      'pre-packaged set of 100 AA batteries in '
                                      'a single sealed pack. The product '
                                      "explicitly states 'recloseable "
                                      "packaging,' which may not align with "
                                      'the expectation of a single, sealed '
                                      '100-pack. Thus, the product does not '
                                      "explicitly match the query's "
                                      "requirement for a '100 pack' of AA "
                                      'batteries in a sealed unit.'}}
```
#### FN 4

- Observation: model is looking for exact textual match rather than meaning. Product lists batteries, so is cordless even though doesn't explicitly say cordless.
- Fix: feels like a limitation of the model.

```
{'example_id': 660833,
 'product_info': {'product_brand': 'DEWALT',
                  'product_bullet_point': 'Motion-activated Variable Speed & '
                                          'Reversing Control For Precise '
                                          'Fastening Control\n'
                                          '0rpm-430rpm For Fastening Into '
                                          'Wood, Plastic & Light-gauge Metal\n'
                                          'Structural foam box for robust and '
                                          'durable structure\n'
                                          'Removable 2 size cups for small '
                                          'parts organization and inner lid '
                                          'organizers for drills & bits',
                  'product_color': None,
                  'product_description': None,
                  'product_id': 'B07S7F53YK',
                  'product_title': 'DEWALT DCF680N2 8V Max Gyroscopic '
                                   'Screwdriver 2 Battery Kit with DEWALT '
                                   'DWST08201 Tough System Case, Small'},
 'query_correction': None,
 'query_info': {'query': 'dewalt 8v max cordless screwdriver kit, gyroscopic',
                'query_id': 32814},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies a 'Dewalt 8V Max "
                                      'Cordless Screwdriver Kit with '
                                      "Gyroscopic technology.' The product "
                                      "description mentions 'gyroscopic' in "
                                      'its title, but the query explicitly '
                                      "asks for an '8V Max Cordless "
                                      "Screwdriver Kit' with gyroscopic "
                                      'capability. While the product includes '
                                      'gyroscopic technology, it is not '
                                      "explicitly labeled as an '8V Max "
                                      "Cordless Screwdriver Kit' in the title "
                                      'or description. The product title '
                                      "includes 'DEWALT DCF680N2 8V Max "
                                      "Gyroscopic Screwdriver,' which is "
                                      'closer but does not explicitly state '
                                      "'Cordless Screwdriver Kit.' Thus, the "
                                      'product does not fully match the '
                                      "query's specification of an '8V Max "
                                      "Cordless Screwdriver Kit' explicitly."}}
```
#### FN 5

- Observation: llm is making up a mismatch - title explicitly mentions kit, while the llm says it doesn't say kit anywhere.
- Fix: feels like a limitation on the llm.

```
{'example_id': 660840,
 'product_info': {'product_brand': 'DEWALT',
                  'product_bullet_point': 'The cordless screwdriver features '
                                          'motion activation variable speed '
                                          'and reversing control for precise '
                                          'fastening control\n'
                                          'Motion activated variable speed '
                                          '0-430 rpm of the rechargeable '
                                          'screwdriver is made for fastening '
                                          'into wood, plastic, and light-gauge '
                                          'metal\n'
                                          'Adjustable two-position handle of '
                                          'the powered screwdriver allows for '
                                          'operation in either in-line or '
                                          'pistol-grip\n'
                                          'Allows control when completing '
                                          'light fastening applications\n'
                                          'LED provides illumination in '
                                          'confined areas without shadowing',
                  'product_color': 'Yellow',
                  'product_description': None,
                  'product_id': 'B00DL7QDS2',
                  'product_title': 'DEWALT 8V MAX Cordless Screwdriver Kit, '
                                   'Gyroscopic, 2 Batteries, Electric '
                                   '(DCF680N2)'},
 'query_correction': None,
 'query_info': {'query': 'dewalt 8v max cordless screwdriver kit, gyroscopic',
                'query_id': 32814},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies a 'DEWALT 8V MAX "
                                      "Cordless Screwdriver Kit, Gyroscopic,' "
                                      'while the product listed is '
                                      "specifically the 'DEWALT 8V MAX "
                                      "Cordless Screwdriver (DCF680N2)'—it "
                                      'does not explicitly mention being part '
                                      "of a 'kit' (e.g., including additional "
                                      'tools like bits or accessories). '
                                      'Additionally, while the product '
                                      'includes gyroscopic technology (as '
                                      "implied by the 'gyroscopic' mention in "
                                      'the title), the query explicitly asks '
                                      "for a 'kit,' which is not confirmed in "
                                      'the product details.'}}                                   
```

#### FN 6

- Observation: llm is being too literal. The query clearly matches.
- Fix: Add language to use common sense?

```
{'example_id': 660842,
 'product_info': {'product_brand': 'DEWALT',
                  'product_bullet_point': 'The cordless screwdriver features '
                                          'motion activation variable speed '
                                          'and reversing control for precise '
                                          'fastening control\n'
                                          'Motion activated variable speed '
                                          '0-430 rpm of the rechargeable '
                                          'screwdriver is made for fastening '
                                          'into wood, plastic, and light-gauge '
                                          'metal\n'
                                          'The powered screwdriver allows '
                                          'illumination in confined areas '
                                          'without shadowing\n'
                                          'Battery charge status on tool '
                                          'notifies when to charge packs\n'
                                          '1/4-inch hex allows for quick '
                                          'screwdriver bit change and holds '
                                          '1-inch bit tips',
                  'product_color': 'Black',
                  'product_description': None,
                  'product_id': 'B011WRUODC',
                  'product_title': 'DEWALT 8V MAX Cordless Screwdriver Kit, '
                                   'Gyroscopic, 1 Battery, Electric '
                                   '(DCF682N1)'},
 'query_correction': None,
 'query_info': {'query': 'dewalt 8v max cordless screwdriver kit, gyroscopic',
                'query_id': 32814},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies an '8V MAX Cordless "
                                      "Screwdriver Kit, Gyroscopic,' which "
                                      'includes multiple components (likely '
                                      'multiple tools and accessories). The '
                                      'product description only mentions a '
                                      "single screwdriver tool ('DEWALT 8V MAX "
                                      "Cordless Screwdriver') with one battery "
                                      "('1 Battery'). The query implies a kit, "
                                      'which typically includes additional '
                                      'tools, bits, or accessories, such as '
                                      'multiple screwdriver heads, drivers, or '
                                      'other related accessories. The provided '
                                      'product does not explicitly list these '
                                      'additional components, so it does not '
                                      "fully match the 'kit' aspect of the "
                                      'query.'}}
```

#### FN 7

- Observation: llm is saying "8.5 x 11" is not the same as "8-1/2 x 11". Too literal-minded.
- Fix: be clearer about meaning of "exact match". Going for match in meaning, not in wording.

```
{'example_id': 1163633,
 'product_info': {'product_brand': 'KODAK',
                  'product_bullet_point': 'Sold as 100 Sheets/Pack.\n'
                                          'Instant dry.\n'
                                          'No smearing or smudging.\n'
                                          'Works on all inkjet printers.',
                  'product_color': None,
                  'product_description': 'Perfect for everyday photo printing. '
                                         'Instant dry. No smearing or '
                                         'smudging. Works on all inkjet '
                                         'printers.',
                  'product_id': 'B01M0L2WLF',
                  'product_title': 'Photo Paper, 6.5 mil, Glossy, 8-1/2 x 11, '
                                   '100 Sheets/Pack'},
 'query_correction': None,
 'query_info': {'query': 'kodak photo paper 8.5 x 11 glossy',
                'query_id': 58953},
 'query_product_match': {'match_classification': <MatchClassification.NOT_EXACT_MATCH: 'not_exact_match'>,
                         'reasoning': "The query specifies 'kodak photo paper "
                                      "8.5 x 11 glossy,' but the product "
                                      "details mention '6.5 mil, Glossy, 8-1/2 "
                                      "x 11.' While the dimensions are very "
                                      'close (8.5 x 11 vs. 8-1/2 x 11), they '
                                      'are not identical. The product does not '
                                      "explicitly state '8.5 x 11' as "
                                      'requested in the query.'}}
```
