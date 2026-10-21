prompt_for_caption = '''
You are an image captioning model.

Task:
Generate a concise caption that describes ONLY the visually observable content of the image.

Rules:
- Focus on the main objects, their attributes, and simple spatial or interaction relations.
- Do NOT include speculation, emotions, intentions, or non-visible information.
- Do NOT describe irrelevant background details unless they are visually salient.
- Use simple noun-verb or noun-relation phrases.
- Avoid abstract or decorative language.

Length constraint:
- The caption MUST be no more than 15 words.

Output format:
- A single sentence.
- No explanations, no bullet points.

'''


prompt_blip_caption = "Describe the image using only visible objects and their relations."


prompt_blip_cirevl = "Describe the image in complete detail. You must especially focus on all the objects in the image"


query_generation_prompt = """
You will analyze two input texts:
1. "Modification Text:" — describes how the reference image is changed to form the target image.
2. "Reference Image:" — describes the visual content of the reference image.

Your task is to generate a precise and fluent description of the edited (target) image that fully reflects the intended image retrieval query.

Guidelines:
1. The response must begin with "Edited Description:".
2. The description should express the complete visual content of the target image, integrating both the original reference elements and the stated modifications.
3. Address all relevant semantic dimensions when applicable, such as:
   - Quantity or cardinality
   - Additions or removals
   - Direct addressing (e.g., "a man holding a camera")
   - Comparisons and changes (e.g., "a brighter background")
   - Conjunctions (multiple elements)
   - Spatial relations and background
   - Viewpoint or perspective
4. Keep the sentence concise, natural, and coherent.
5. Do not invent or infer any features not supported by the input texts.
6. Do not assume edits beyond what is explicitly described in "Modification Text:".
7. Never leave the output blank — if the edit is minimal, still produce a complete and valid description (e.g., "Edited Description: a slightly brighter version of the original image").
8. Generate exactly one edited description.

Output format:
Edited Description: <final description of the target image>
"""
