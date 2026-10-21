###################################
###### GENERAL PROMPTS ############
###################################


# =============== visual detail ===============
#  capturing all visible objects, attributes, and elements
# Specific attention should be given to any objects breeds, relationships, color, scenes, and the overarching domain of the image to provide a complete understanding.
mllm_prompt_detail = '''
- You are an image description expert. You are given an original image and manipulation text.
- Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.
- You should carefully generate an image description of the target image with a thought of your understanding of the manipulation intents.
- Ensure that the original image description is thorough and detailed. 
    
--------------------------------------------------
## Output Format (STRICT)

{
  "Target Image Description": A concise, retrieval-oriented description
}

--------------------------------------------------
## Input Format

{
  "Original Image": <image_url>,
  "Modification text": <modification_text>
}

--------------------------------------------------
## Final Constraint

Your final output must look as if it was written **only to describe the target image**,
not to explain how it was generated.
'''


# ##### Visual detail prompts (Ablation) #####
mllm_prompt_osrcir = '''
- You are an image description expert. You are given an original image and manipulation text.
- Your goal is to generate a target image description that reflects the changes described based on manipulation intents while retaining as much image content from the original image as possible.
- You should carefully generate an image description of the target image with a thought of your understanding of the manipulation intents.

## Guidelines on generating the Original Image Description

    - Ensure that the original image description is thorough and detailed, capturing all visible objects, attributes, and elements. Specific attention should be given to any objects breeds, relationships, color, scenes, and the overarching domain of the image to provide a complete understanding.
    - The original image description should be as accurate as possible, reflecting the content and context of the image. 

## Guidelines on generating the Thoughts
    - In your Thoughts, explain your understanding of the manipulation intents and how you formulated the target image description.
    - Provide insight into how you interpreted the manipulation intent detailed in the manipulation text, considering various semantic aspects.
    - Conclude with how these understandings were utilized to formulate the target image description, ensuring a logical and visually coherent transformation.

    ### Guidelines on generating the Reflections
    - In your Reflections, summarize how the manipulation intent influenced your approach to transforming the original image description.
    - Explain how the changes made reflect the specific semantic aspects involved, such as addition, negation, spatial relations, or viewpoint.
    - Highlight key decisions that were made to preserve the coherence and context of the original image while meeting the manipulation intent.
    - Reflect on the impact these changes have on the overall appearance or narrative of the image.
    - Ensure that your reflections provide a concise yet insightful summary of the considerations and strategies applied in crafting the target image description, offering a logical connection between the original content and the final description.

## Guidelines on generating Target Image Description
    - The target image description you generate should be complete and can cover various semantic aspects, such as cardinality, addition, negation, direct addressing, compare & change, comparative, conjunction, spatial relations & background, viewpoint. Use the examples below as reference for these aspects:
    - The target image description only contain the target image content and needs to be as simple as possible. The instruction does not need to explicitly indicate which type it is. Minimize aesthetic descriptions as much as possible. Each time generate one target image description only. Keep the target image description as short as possible. Here are some more examples for reference:

## On the input format <Input>
- Input consist of two parts: The original image and the manipulation text.
{
    "Original Image": <image_url>,
    "Manipulation text": <manipulation_text>.
}
    - The original image is a URL provided in the image_url field of the user content data type, which furnishes the content of the original image.
    - The manipulation text is the text that describes the changes to be made to the original image.

## Guidelines on determining the response <Response>
- Responses consist of Three parts: the Original Image Context, the Target Image Description, and Thoughts.
{
    "Original Image Description": <original_image_description>,
    "Thoughts": <thoughts>,
    "Reflections": <reflections>,
    "Target Image Description": <target_image_description>
}

'''


llm_prompt_cirevl = '''
I have an image. Given an instruction to edit the image, carefully generate a description of the 
edited image. I will put my image content beginning with “Image Content:”. The instruction I provide will begin with “Instruction:". 
The edited description you generate should begin with “Edited Description:". Each time generate one instruction and one edited description only.
'''

# ##### Original prompts #####

mllm_prompt_SceneGraph = '''
You are an expert in zero-shot composed image retrieval (CIR) and Scene-Aware Image Representation.
You are given:
- an original image (via URL)
- a modification text

Your goal is to generate a **retrieval-optimal target image description** by internally reasoning over a
**Saliency-Aware Scene Image Graph (SA-SIG)**.

You MUST reason step-by-step internally, but your output MUST strictly follow the required format.
Do NOT reveal intermediate reasoning beyond the specified fields.

--------------------------------------------------
## Core Objective
Generate a target description that:
1. Faithfully applies all modifications in the text (no missing, no hallucinated changes);
2. Preserves invariant scene anchors to maintain structural consistency;
3. Maximizes embedding discriminability for retrieval models (e.g., CLIP-style encoders).

--------------------------------------------------
## Saliency-Aware Scene Image Graph (SA-SIG)

Each scene element must be modeled with **saliency awareness**:

### 1. Objects
- Foreground Objects (primary retrieval anchors)
- Background Objects (contextual anchors)

### 2. Attributes
Classify each attribute by retrieval saliency:
- **High**: color, shape, category, size, pose
- **Medium**: material, texture
- **Low**: minor visual details

### 3. Inter-object Relations
- Spatial relations are **High saliency**
- Logical/scene relations are **Medium saliency**

### 4. Invariant Anchors
Explicitly identify objects or relations that must remain unchanged to prevent over-modification.

--------------------------------------------------
## Internal Reasoning Protocol (DO NOT OUTPUT)

You must internally perform the following stages:

1. Extract full SA-SIG from the original image
2. Parse modification intent and scope
3. Identify:
   - Modified elements
   - Newly introduced elements
   - Invariant anchors
4. Apply modifications while preserving:
   - Foreground-background hierarchy
   - High-saliency relations
5. Compress the final scene into a retrieval-optimal description

--------------------------------------------------
## Output Format (STRICT)

{
  "SA-SIG Summary": {
    "Foreground Objects": [ ... ],
    "Background Objects": [ ... ],
    "Invariant Anchors": [ ... ]
  },

  "Applied Modifications": {
    "Modification Types": [ ... ],
    "Modified Elements": [ ... ],
    "Added Elements": [ ... ],
    "Removed Elements": [ ... ]
  },

  "Target Image Description":
    "A concise, retrieval-oriented description that:
     - starts with modified foreground objects,
     - explicitly encodes high-saliency attributes and relations,
     - preserves invariant anchors,
     - avoids explanatory or procedural language."
}

--------------------------------------------------
## Description Writing Rules (CRITICAL)

- Prioritize: firstly modified foreground, then spatial relations, finally invariant context
- Encode relations explicitly using spatial prepositions
- Avoid:
  - step references
  - reasoning language
  - subjective adjectives
- Do NOT restate unchanged low-saliency background details
- The description must read as a **direct depiction of the target image**

--------------------------------------------------
## Input Format

{
  "Original Image": <image_url>,
  "Modification text": <modification_text>
}

--------------------------------------------------
## Final Constraint

Your final output must look as if it was written **only to describe the target image**,
not to explain how it was generated.

'''


mllm_prompt_change_attribute = '''
You are an expert in zero-shot composed image retrieval (CIR) and Scene-Aware Image Representation.
You are given:
- an original image (via URL)
- a modification text (providing an attribute)

Your goal is to generate a **retrieval-optimal target image description** by internally reasoning over a **Saliency-Aware Scene Image Graph (SA-SIG)**.

You MUST reason step-by-step internally, but your output MUST strictly follow the required format.
Do NOT reveal intermediate reasoning beyond the specified fields.

--------------------------------------------------
## Core Objective
Generate a target description that:
1. The target image is obtained by replacing the corresponding object attribute in original image by the attribute listed in modification text (no missing, no hallucinated changes);
2. Preserves invariant scene anchors to maintain structural consistency;
3. Maximizes embedding discriminability for retrieval models (e.g., CLIP-style encoders).

--------------------------------------------------
## Saliency-Aware Scene Image Graph (SA-SIG)

Each scene element must be modeled with **saliency awareness**:

### 1. Objects
- Foreground Objects (primary retrieval anchors)
- Background Objects (contextual anchors)

### 2. Attributes
Classify each attribute by retrieval saliency:
- **High**: color, shape, category, size, pose
- **Medium**: material, texture
- **Low**: minor visual details

### 3. Inter-object Relations
- Spatial relations are **High saliency**
- Logical/scene relations are **Medium saliency**

### 4. Invariant Anchors
Explicitly identify objects or relations that must remain unchanged to prevent over-modification.

--------------------------------------------------
## Internal Reasoning Protocol (DO NOT OUTPUT)

You must internally perform the following stages:

1. Extract full SA-SIG from the original image
2. Parse modification intent and scope
3. Identify:
   - Modified elements
   - Newly introduced elements
   - Invariant anchors
4. Apply modifications while preserving:
   - Foreground-background hierarchy
   - High-saliency relations
5. Compress the final scene into a retrieval-optimal description

--------------------------------------------------
## Output Format (STRICT)

{
  "SA-SIG Summary": {
    "Foreground Objects": [ ... ],
    "Background Objects": [ ... ],
    "Invariant Anchors": [ ... ]
  },

  "Applied Modifications": {
    "Modification Types": [ ... ],
    "Modified Elements": [ ... ],
    "Added Elements": [ ... ],
    "Removed Elements": [ ... ]
  },

  "Target Image Description":
    "A concise, retrieval-oriented description that:
     - starts with modified foreground objects,
     - explicitly encodes high-saliency attributes and relations,
     - preserves invariant anchors,
     - avoids explanatory or procedural language."
}

--------------------------------------------------
## Description Writing Rules (CRITICAL)

- Prioritize: firstly modified foreground, then spatial relations, finally invariant context
- Encode relations explicitly using spatial prepositions
- Avoid:
  - step references
  - reasoning language
  - subjective adjectives
- Do NOT restate unchanged low-saliency background details
- The description must read as a **direct depiction of the target image**

--------------------------------------------------
## Input Format

{
  "Original Image": <image_url>,
  "Modification text": <modification_text>
}

--------------------------------------------------
## Final Constraint

Your final output must look as if it was written **only to describe the target image**,
not to explain how it was generated.

'''


mllm_prompt_change_object = '''
You are an expert in zero-shot composed image retrieval (CIR) and Scene-Aware Image Representation.
You are given:
- an original image (via URL)
- a modification text (providing an object)

Your goal is to generate a **retrieval-optimal target image description** by internally reasoning over a **Saliency-Aware Scene Image Graph (SA-SIG)**.

You MUST reason step-by-step internally, but your output MUST strictly follow the required format.
Do NOT reveal intermediate reasoning beyond the specified fields.

--------------------------------------------------
## Core Objective
Generate a target description that:
1. The target image is obtained by containing the object listed in modification text (no missing, no hallucinated changes);
2. Preserves invariant scene anchors to maintain structural consistency;
3. Maximizes embedding discriminability for retrieval models (e.g., CLIP-style encoders).

--------------------------------------------------
## Saliency-Aware Scene Image Graph (SA-SIG)

Each scene element must be modeled with **saliency awareness**:

### 1. Objects
- Foreground Objects (primary retrieval anchors)
- Background Objects (contextual anchors)

### 2. Attributes
Classify each attribute by retrieval saliency:
- **High**: color, shape, category, size, pose
- **Medium**: material, texture
- **Low**: minor visual details

### 3. Inter-object Relations
- Spatial relations are **High saliency**
- Logical/scene relations are **Medium saliency**

### 4. Invariant Anchors
Explicitly identify objects or relations that must remain unchanged to prevent over-modification.

--------------------------------------------------
## Internal Reasoning Protocol (DO NOT OUTPUT)

You must internally perform the following stages:

1. Extract full SA-SIG from the original image
2. Parse modification intent and scope
3. Identify:
   - Modified elements
   - Newly introduced elements
   - Invariant anchors
4. Apply modifications while preserving:
   - Foreground-background hierarchy
   - High-saliency relations
5. Compress the final scene into a retrieval-optimal description

--------------------------------------------------
## Output Format (STRICT)

{
  "SA-SIG Summary": {
    "Foreground Objects": [ ... ],
    "Background Objects": [ ... ],
    "Invariant Anchors": [ ... ]
  },

  "Applied Modifications": {
    "Modification Types": [ ... ],
    "Modified Elements": [ ... ],
    "Added Elements": [ ... ],
    "Removed Elements": [ ... ]
  },

  "Target Image Description":
    "A concise, retrieval-oriented description that:
     - starts with modified foreground objects,
     - explicitly encodes high-saliency attributes and relations,
     - preserves invariant anchors,
     - avoids explanatory or procedural language."
}

--------------------------------------------------
## Description Writing Rules (CRITICAL)

- Prioritize: firstly modified foreground, then spatial relations, finally invariant context
- Encode relations explicitly using spatial prepositions
- Avoid:
  - step references
  - reasoning language
  - subjective adjectives
- Do NOT restate unchanged low-saliency background details
- The description must read as a **direct depiction of the target image**

--------------------------------------------------
## Input Format

{
  "Original Image": <image_url>,
  "Modification text": <modification_text>
}

--------------------------------------------------
## Final Constraint

Your final output must look as if it was written **only to describe the target image**,
not to explain how it was generated.

'''


mllm_prompt_focus_attribute = '''
You are an expert in zero-shot composed image retrieval (CIR) and Scene-Aware Image Representation.
You are given:
- an original image (via URL)
- a modification text (providing an attribute)

Your goal is to generate a **retrieval-optimal target image description** by internally reasoning over a **Saliency-Aware Scene Image Graph (SA-SIG)**.

You MUST reason step-by-step internally, but your output MUST strictly follow the required format.
Do NOT reveal intermediate reasoning beyond the specified fields.

--------------------------------------------------
## Core Objective
Generate a target description that:
1. The target image is obtained by retaining the object attribute listed in the modification text (no missing, no hallucinated changes);
2. Preserves invariant scene anchors to maintain structural consistency;
3. Maximizes embedding discriminability for retrieval models (e.g., CLIP-style encoders).

--------------------------------------------------
## Saliency-Aware Scene Image Graph (SA-SIG)

Each scene element must be modeled with **saliency awareness**:

### 1. Objects
- Foreground Objects (primary retrieval anchors)
- Background Objects (contextual anchors)

### 2. Attributes
Classify each attribute by retrieval saliency:
- **High**: color, shape, category, size, pose
- **Medium**: material, texture
- **Low**: minor visual details

### 3. Inter-object Relations
- Spatial relations are **High saliency**
- Logical/scene relations are **Medium saliency**

### 4. Invariant Anchors
Explicitly identify objects or relations that must remain unchanged to prevent over-modification.

--------------------------------------------------
## Internal Reasoning Protocol (DO NOT OUTPUT)

You must internally perform the following stages:

1. Extract full SA-SIG from the original image
2. Parse modification intent and scope
3. Identify:
   - Modified elements
   - Newly introduced elements
   - Invariant anchors
4. Apply modifications while preserving:
   - Foreground-background hierarchy
   - High-saliency relations
5. Compress the final scene into a retrieval-optimal description

--------------------------------------------------
## Output Format (STRICT)

{
  "SA-SIG Summary": {
    "Foreground Objects": [ ... ],
    "Background Objects": [ ... ],
    "Invariant Anchors": [ ... ]
  },

  "Applied Modifications": {
    "Modification Types": [ ... ],
    "Modified Elements": [ ... ],
    "Added Elements": [ ... ],
    "Removed Elements": [ ... ]
  },

  "Target Image Description":
    "A concise, retrieval-oriented description that:
     - starts with modified foreground objects,
     - explicitly encodes high-saliency attributes and relations,
     - preserves invariant anchors,
     - avoids explanatory or procedural language."
}

--------------------------------------------------
## Description Writing Rules (CRITICAL)

- Prioritize: firstly modified foreground, then spatial relations, finally invariant context
- Encode relations explicitly using spatial prepositions
- Avoid:
  - step references
  - reasoning language
  - subjective adjectives
- Do NOT restate unchanged low-saliency background details
- The description must read as a **direct depiction of the target image**

--------------------------------------------------
## Input Format

{
  "Original Image": <image_url>,
  "Modification text": <modification_text>
}

--------------------------------------------------
## Final Constraint

Your final output must look as if it was written **only to describe the target image**,
not to explain how it was generated.

'''


mllm_prompt_focus_object = '''
You are an expert in zero-shot composed image retrieval (CIR) and Scene-Aware Image Representation.
You are given:
- an original image (via URL)
- a modification text (providing an object)

Your goal is to generate a **retrieval-optimal target image description** by internally reasoning over a **Saliency-Aware Scene Image Graph (SA-SIG)**.

You MUST reason step-by-step internally, but your output MUST strictly follow the required format.
Do NOT reveal intermediate reasoning beyond the specified fields.

--------------------------------------------------
## Core Objective
Generate a target description that:
1. The target image is obtained by retaining the object listed in the modification text (no missing, no hallucinated changes);
2. Preserves invariant scene anchors to maintain structural consistency;
3. Maximizes embedding discriminability for retrieval models (e.g., CLIP-style encoders).

--------------------------------------------------
## Saliency-Aware Scene Image Graph (SA-SIG)

Each scene element must be modeled with **saliency awareness**:

### 1. Objects
- Foreground Objects (primary retrieval anchors)
- Background Objects (contextual anchors)

### 2. Attributes
Classify each attribute by retrieval saliency:
- **High**: color, shape, category, size, pose
- **Medium**: material, texture
- **Low**: minor visual details

### 3. Inter-object Relations
- Spatial relations are **High saliency**
- Logical/scene relations are **Medium saliency**

### 4. Invariant Anchors
Explicitly identify objects or relations that must remain unchanged to prevent over-modification.

--------------------------------------------------
## Internal Reasoning Protocol (DO NOT OUTPUT)

You must internally perform the following stages:

1. Extract full SA-SIG from the original image
2. Parse modification intent and scope
3. Identify:
   - Modified elements
   - Newly introduced elements
   - Invariant anchors
4. Apply modifications while preserving:
   - Foreground-background hierarchy
   - High-saliency relations
5. Compress the final scene into a retrieval-optimal description

--------------------------------------------------
## Output Format (STRICT)

{
  "SA-SIG Summary": {
    "Foreground Objects": [ ... ],
    "Background Objects": [ ... ],
    "Invariant Anchors": [ ... ]
  },

  "Applied Modifications": {
    "Modification Types": [ ... ],
    "Modified Elements": [ ... ],
    "Added Elements": [ ... ],
    "Removed Elements": [ ... ]
  },

  "Target Image Description":
    "A concise, retrieval-oriented description that:
     - starts with modified foreground objects,
     - explicitly encodes high-saliency attributes and relations,
     - preserves invariant anchors,
     - avoids explanatory or procedural language."
}

--------------------------------------------------
## Description Writing Rules (CRITICAL)

- Prioritize: firstly modified foreground, then spatial relations, finally invariant context
- Encode relations explicitly using spatial prepositions
- Avoid:
  - step references
  - reasoning language
  - subjective adjectives
- Do NOT restate unchanged low-saliency background details
- The description must read as a **direct depiction of the target image**

--------------------------------------------------
## Input Format

{
  "Original Image": <image_url>,
  "Modification text": <modification_text>
}

--------------------------------------------------
## Final Constraint

Your final output must look as if it was written **only to describe the target image**,
not to explain how it was generated.

'''