# Gallery index (`assets/gallery.*`)

Tiles are listed row by row (top-left first; the centre poster is `P`). Every tile was sampled with **4 steps, no CFG** by a released checkpoint of this repository (images) or of `fewsteprl-wan21` (videos). Prompts and seeds are curated: per prompt, among 8 seeds, using the paper's reward models (HPSv2, PickScore, aesthetic, ImageReward) plus visual inspection, which includes a check for the faint 16-px patch-lattice pattern that some 4-step samples carry (image tiles with a visible lattice were replaced by a clean sample of the same prompt); the centre poster is one of the 257 posters, out of 2,304 sampled (12 poster prompts × 48 seeds × 4 trainers), in which PP-OCRv5 reads both "FewStepRL" and "Hunyuan", picked for clean anatomy and lettering.

Reproduce an image tile bit-exactly from the repository root:

```bash
python inference.py scripts/eval/configs/sample_sd35m_fewstep.yaml --prompt "<prompt>" \
    --override seed=<seed> batch_size=1 num_stages=3 stage3_lora_path=<checkpoint>
```

and a video tile from the `fewsteprl-wan21` root:

```bash
python inference.py scripts/eval/configs/sample_wan.yaml --prompt "<prompt>" \
    --override seed=<seed> mode=meanflow_nft meanflow_nft_path=<checkpoint>
```

| tile | trainer | kind | seed | checkpoint | prompt |
|---|---|---|---|---|---|
| A | DMDNFT | SD3.5-M image, 512² | 1 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A giant glowing tree of life on a floating island at night, bioluminescent leaves, fireflies, a starry galaxy behind it, magical atmosphere. |
| B | ReFL | Wan2.1 video, 81×480×832 | 0 | `checkpoints/reflmx/checkpoint-250/generator_ema.pt` | Cherry blossom petals falling on an old Kyoto street at dusk with glowing lanterns, people walking with umbrellas, cinematic. |
| C | MeanFlowNFT | SD3.5-M image, 512² | 1 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A snowy owl in flight with wings fully spread against falling snow, piercing yellow eyes, award-winning wildlife photograph. |
| D | FewStepGRPO | Wan2.1 video, 81×480×832 | 0 | `checkpoints/fewstepgrpo/checkpoint-250/generator_ema.pt` | Hundreds of blue morpho butterflies fluttering through a sunlit rainforest, light rays through the canopy, magical slow motion. |
| E | DMDNFT | SD3.5-M image, 512² | 6 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A woman in a flowing golden silk dress standing on a sand dune as the wind lifts the fabric into the sky, epic fashion photography at sunset. |
| F | DMDNFT | SD3.5-M image, 512² | 6 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A vibrant octopus with glowing patterns curling its tentacles around a coral reef, sunbeams from the surface, underwater photography. |
| G | DMDNFT | SD3.5-M image, 512² | 5 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A tiny ruby-throated hummingbird hovering next to a glowing pink orchid, wings in motion, macro photography, dark green background. |
| H | FewStepGRPO | SD3.5-M image, 512² | 0 | `checkpoints/fewstepgrpo_mix/checkpoint-300/generator_ema.pt` | A surfer riding inside the barrel of a giant turquoise wave, sunlight shining through the water, action sports photography. |
| I | MeanFlowAWM | SD3.5-M image, 512² | 5 | `checkpoints/meanflowawm_mix/checkpoint-400/generator_ema.pt` | A cute red fox wearing a knitted scarf sipping tea at a tiny table in an autumn forest, falling leaves, storybook illustration. |
| J | MeanFlowNFT | SD3.5-M image, 512² | 1 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A curious tabby kitten looking into an ornate vintage mirror that reflects a majestic lion, warm cozy room, whimsical photography. |
| K | DMDNFT | SD3.5-M image, 512² | 2 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A breathtaking futuristic megacity at night with flying cars, holographic billboards and neon skyscrapers piercing the clouds, cyberpunk masterpiece. |
| L | MeanFlowNFT | Wan2.1 video, 81×480×832 | 0 | `checkpoints/meanflownft/checkpoint-250/generator_ema.pt` | A phoenix made of fire spreading its wings and rising into the night sky, embers and sparks swirling around it, cinematic slow motion. |
| M | DMDNFT | SD3.5-M image, 512² | 0 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | Endless lavender fields in Provence at sunset with a lone tree and a stone farmhouse, warm golden light, landscape photography. |
| N | DMDNFT | SD3.5-M image, 512² | 5 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A powerful waterfall plunging into a turquoise canyon with a vivid rainbow in the mist, lush green cliffs, epic landscape photography. |
| Q | DMDNFT | SD3.5-M image, 512² | 0 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A lone explorer with a lantern standing inside a gigantic cave of glowing purple and teal crystals, reflections on an underground lake, breathtaking scale. |
| R | MeanFlowNFT | SD3.5-M image, 512² | 3 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | An erupting volcano at night with lava rivers and a volcanic lightning storm in the ash cloud, dramatic nature photography. |
| S | MeanFlowAWM | SD3.5-M image, 512² | 0 | `checkpoints/meanflowawm_mix/checkpoint-400/generator_ema.pt` | Misty karst mountains along the Li River at dawn, a fisherman on a bamboo raft with cormorants, traditional Chinese landscape, cinematic. |
| T | MeanFlowNFT | SD3.5-M image, 512² | 7 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A giant mech samurai with glowing red visor kneeling in a field of cherry blossoms, petals swirling in the wind, cinematic sci-fi art. |
| P | DMDNFT | SD3.5-M image, 512² | 15 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | An epic poster with the large title "FewStepRL" at the top and the small text "Hunyuan" at the bottom, an ancient temple on a misty mountain peak above a sea of clouds at sunrise, glowing lanterns, cinematic. |
| U | MeanFlowNFT | SD3.5-M image, 512² | 6 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A young monk in orange robes walking up ancient stone stairs to a mountain temple in the clouds, cinematic travel photography. |
| V | DMDNFT | SD3.5-M image, 512² | 2 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A red sandstone arch under the glowing Milky Way in the desert, starry night, astrophotography. |
| W | FewStepOPSD | Wan2.1 video, 81×480×832 | 0 | `checkpoints/meanflowopsd/checkpoint-250/generator_ema.pt` | A giant whale swimming slowly through pink sunset clouds above a small seaside town, dreamy surreal cinematic shot. |
| X | FewStepGRPO | SD3.5-M image, 512² | 7 | `checkpoints/fewstepgrpo_mix/checkpoint-300/generator_ema.pt` | A translucent jellyfish whose bell contains a tiny swirling galaxy, floating in the dark ocean, surreal glowing art. |
| Y | DMDNFT | SD3.5-M image, 512² | 2 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A Bengal tiger walking through shallow water in a misty jungle at golden hour, splashes frozen in motion, wildlife photography, ultra sharp. |
| Z | MeanFlowNFT | SD3.5-M image, 512² | 1 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A black stallion galloping along the shoreline at sunset, mane flying, water splashing, golden backlight, cinematic photography. |
| a | DMDNFT | SD3.5-M image, 512² | 6 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A cracked open geode revealing a tiny glowing galaxy inside, resting on moss, macro photography, magical. |
| b | DMDNFT | SD3.5-M image, 512² | 5 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | An elegant crystal perfume bottle surrounded by floating rose petals and golden droplets, soft pink light, luxury advertising photography. |
| c | MeanFlowNFT | SD3.5-M image, 512² | 2 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | An exquisite platter of sushi and sashimi on a black slate with gold leaf, chopsticks, dramatic overhead food photography. |
| d | MeanFlowAWM | SD3.5-M image, 512² | 5 | `checkpoints/meanflowawm_mix/checkpoint-400/generator_ema.pt` | A close-up portrait of a beautiful cyborg woman with intricate porcelain and gold mechanical details on her face, soft studio light, hyperrealistic. |
| e | DMDNFT | SD3.5-M image, 512² | 0 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A fairytale castle growing out of the pages of an open old book, tiny glowing windows, magical dust, surreal photography. |
| f | DMDNFT | Wan2.1 video, 81×480×832 | 0 | `checkpoints/dmdnft/checkpoint-250/generator_ema.pt` | Thousands of glowing paper lanterns floating up into the night sky over a calm lake, reflections on the water, magical atmosphere. |
| g | MeanFlowNFT | SD3.5-M image, 512² | 6 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A small train crossing a sea at sunset with clouds reflected in the water, Studio Ghibli anime style, beautiful painted background. |
| h | MeanFlowAWM | Wan2.1 video, 81×480×832 | 0 | `checkpoints/meanflowawm/checkpoint-250/generator_ema.pt` | Vivid green and purple northern lights dancing over a snowy mountain lake, stars twinkling, time-lapse. |
| i | DMDNFT | SD3.5-M image, 512² | 0 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A glass terrarium containing a miniature waterfall, tiny trees and a little wooden house, soft window light, macro photography. |
| j | DMDNFT | SD3.5-M image, 512² | 3 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A white porcelain rabbit figurine with delicate blue floral patterns on a dark velvet background, museum photography. |
| k | MeanFlowNFT | SD3.5-M image, 512² | 3 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A stained glass window depicting a phoenix and a dragon in vivid jewel colors with sunlight streaming through, cathedral interior. |
| l | DMDNFT | SD3.5-M image, 512² | 5 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A molten chocolate lava cake with vanilla ice cream and fresh raspberries, chocolate flowing out, close-up dessert photography. |
| m | DMDNFT | SD3.5-M image, 512² | 1 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A luxury mechanical watch with an open skeleton dial showing intricate golden gears, dramatic studio lighting on black, product photography. |
| n | MeanFlowAWM | SD3.5-M image, 512² | 7 | `checkpoints/meanflowawm_mix/checkpoint-400/generator_ema.pt` | An isometric 3D diorama of a cozy floating island village with windmills, waterfalls and tiny people, pastel colors, cute render. |
| o | MeanFlowNFT | SD3.5-M image, 512² | 1 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | An infinite fractal spiral of glowing golden and blue patterns, mathematical beauty, ultra detailed digital art. |
| p | MeanFlowNFT | SD3.5-M image, 512² | 7 | `checkpoints/meanflownft_mix/checkpoint-400/generator_ema.pt` | A portrait of a regal cat in the style of Gustav Klimt, golden ornamental patterns and mosaic background. |
| q | DMDNFT | SD3.5-M image, 512² | 0 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A boy sitting on a crescent moon fishing for stars above a sleeping city, dreamy children's book illustration. |
| r | DMDNFT | SD3.5-M image, 512² | 3 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A white peacock displaying its magnificent tail feathers in a sunlit garden, delicate lace-like plumage, bokeh, elegant photography. |
| s | DMDNFT | SD3.5-M image, 512² | 7 | `checkpoints/dmdnft_mix/checkpoint-1000/generator_ema.pt` | A glowing light bulb containing a tiny thriving forest and a waterfall, on a dark wooden table, surreal macro photography. |
| t | MeanFlowAWM | SD3.5-M image, 512² | 5 | `checkpoints/meanflowawm_mix/checkpoint-400/generator_ema.pt` | A grand piano on a cliff edge with a waterfall pouring out of it into the clouds, surreal fantasy photography at sunset. |
| u | FewStepGRPO | SD3.5-M image, 512² | 2 | `checkpoints/fewstepgrpo_mix/checkpoint-300/generator_ema.pt` | A colossal ring-shaped space station orbiting a blue gas giant, a spaceship approaching, sun flare, photorealistic sci-fi render. |
