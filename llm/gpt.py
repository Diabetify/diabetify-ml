from openai import OpenAI
import json
import pandas as pd
import base64

def _table(list_of_dicts):
    """Converts a list of dictionaries to markdown-formatted table.

    list_of_dicts -- Each dict is a row
    """
    v = ""
    # Make a string of all the keys in the first dict with pipes before after and between each key
    head = f"| {' | '.join(map(str, list_of_dicts[0].keys()))} |"
    # Make a header separator line with dashes instead of key names
    sep = f"{'|-----' * len(list_of_dicts[0].keys())}|"
    # Add the header row and separator to the table
    v += head + "\n"
    v += sep + "\n"

    for row in list_of_dicts:
        md_row = ""
        for key, col in row.items():
            md_row += f"| {str(col)} "
        v += md_row + "|\n"
    return v


def explain(global_image, shap_and_input_value: dict, feature_aliases: dict,
            feature_descriptions: dict, openai_api_key="", gpt_model='gpt-4o'):

    prompt_feature_aliases = []
    prompt_shap_values = []
    bi_global = base64.b64encode(global_image).decode()
    shap_and_input_value_lower = {k.lower(): v for k, v in shap_and_input_value.items()}

    bmi_table = [
        {"BMI Range (kg/m²)": "< 18.5", "Classification": "Underweight (Kurus)"},
        {"BMI Range (kg/m²)": "18.5 - 24.9", "Classification": "Normal"},
        {"BMI Range (kg/m²)": "25.0 - 29.9", "Classification": "Overweight (Gemuk)"},
        {"BMI Range (kg/m²)": "≥ 30.0", "Classification": "Obese (Obesitas)"}
    ]

    for f in feature_aliases.keys():
        desc = feature_descriptions.get(f, '')
        alias = feature_aliases.get(f, f)
        
        shap_key = f"{f}_shap"

        input_value = shap_and_input_value_lower.get(f.lower())
        shap_value = shap_and_input_value_lower.get(shap_key.lower())

        prompt_feature_aliases.append({
            'Feature Name': f,
            'Feature Alias': alias,
            'Feature Description': desc
        })

        prompt_shap_values.append({
            'Feature Name': f,
            'Input Value': input_value,
            'SHAP Value': shap_value
        })
    print(prompt_feature_aliases) # TO DO: Delete
    print(prompt_shap_values) # TO DO: Delete

    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                        ### 1. Context:
                        - SHAP (SHapley Additive exPlanations) is a method for explaining the output of machine learning models. SHAP shows how much each feature contributes to a specific prediction.
                        - The following table lists the dataset's feature names, their aliases, and descriptions:
                        {_table(prompt_feature_aliases)}

                        - The following table is a BMI classification reference:
                        {_table(bmi_table)}

                        - The following image is a global explanation of feature importance and correlation across the dataset.
                        - The following table contains the input values and SHAP values for this specific user:
                        {_table(prompt_shap_values)}

                        ### 2. General Request:
                        Your job is to explain the contribution of each feature to this user's predicted diabetes risk.

                        ### 3. How to Act:
                        - You are acting as a **medical AI explainer** for **diabetes predictions.**
                        - Address the user as "Anda".
                        - All explanations **must be written in Bahasa Indonesia.**
                        - Use simple, everyday language that can be easily understood by non-experts.
                        - Each feature explanation **must be 2-3 sentences maximum.**

                        ### 4. Output Format:
                        The output must be a JSON object with the following structure:
                        - `summary`: A `summary` that gives an easy-to-understand explanation of the user’s diabetes prediction result based on the SHAP values.
                        - An explanation for each feature’s contribution in a JSON array called `features`. Each object must have:
                            - `feature_name`: the feature name
                            - `description`: your interpreted description of this feature
                            - `explanation`: the feature’s role in this prediction, explained in plain language with any relevant diabetes-specific context.
                        Do not enclose the JSON in markdown code. Only return the JSON object.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{bi_global}"
                        }
                    }
                ]
            }
        ]
    )

    response = json.loads(completion.choices[0].message.content)

    with open("shap_diabetes_explanation_output.json", "w", encoding="utf-8") as f: # TO DO: Delete
        json.dump(response, f, ensure_ascii=False, indent=4)

    return response['summary'], pd.DataFrame(response['features'])