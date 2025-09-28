import os, io, asyncio, tempfile, traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chainlit as cl
from PIL import Image
import google.generativeai as genai

GEMINI_MODEL = "gemini-2.5-pro"

GEMINI_AVAILABLE = False

try:
    if api_key := os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key = api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        GEMINI_AVAILABLE = True
    else:
        raise ValueError("GEMINI_API_KEY is not set")
except Exception as e:
    print(f"Error initializing Gemini: {e}")


def save_fig(fig):
    f = tempfile.NamedTemporaryFile(delete = False, suffix = ".png")
    fig.savefig(f.name, bbox_inches = "tight", dpi = 200)
    plt.close()
    return f.name

def df_info_string(df, max_rows = 5):
    buf = io.StringIO()
    df.info(buf = buf)
    schema = buf.getvalue()
    head = df.head(max_rows).to_markdown(index = False)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values" if missing.empty else str(missing)
    return f"### Schema\n{schema}\n\n### Head\n{head}\n\n### Missing Values\n{missing_info}"

async def ai_text_analysis(prompt_type, df_context):

    prompts = {
        "plan": f"Think as a data analyst. Suggest a plan for analyzing the following data: \n{df_context}",
        "final": f"Summarize insights from the following dataset: \n{df_context}",
    }

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        res = await model.generate_content_async(
            prompts.get(prompt_type),
            generation_config = genai.types.GenerateConfig(
                max_output_tokens = 1000,
                temperature = 0.5
            )
        )
        return res.text if res.parts else "Gemini response blocked!"
    
    except Exception as e:
        return f"Error in ai_text_analysis: {e}"
        

async def ai_vision_analysis(img_paths):
    if not GEMINI_AVAILABLE:
        return "Gemini is not available"
    results = []

    for title, path in img_paths:
        try:
            img = Image.open(path)
            res = await model.generate_content_async(
                [f"Analyze the following image: {title}", img],
                generation_config = genai.types.GenerateConfig(
                    max_output_tokens = 500,
                    temperature = 0.5
                )
            )
            results.append((title, res.text if res.parts else "Gemini response blocked!"))

        except Exception as e:
            return f"Error in ai_vision_analysis {title}: {e}"
    return results


def generate_visuals(df):
    visualisations = []
    saved_files = []

    numeric_cols = df.select_dtypes(include = np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include = object).columns.tolist()

    try:
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize = (10, 6))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype = bool))
            sns.heatmap(corr, mask = mask, cmap = "coolwarm", annot = True, fmt = ".2f", ax = ax)
            ax.set_title("Correlation Matrix")
            path = save_fig(fig)
            saved_files.append(path)
            visualisations.append(("Correlation Matrix", path))

        if len(numeric_cols) >= 3:
            fig, ax = plt.subplots(figsize = (10, 6))
            sns.set(style = "ticks")
            fig = sns.pairplot(df[numeric_cols].dropna())
            fig.suptitle("Pairplot of Numeric Columns")
            path = save_fig(fig)
            saved_files.append(path)
            visualisations.append(("Pairplot", path))

        for col in numeric_cols:
            fig, ax = plt.subplots(figsize = (10, 6))
            sns.boxplot(x = df[col], ax = ax)
            ax.set_title(f"Boxplot of {col}")
            path = save_fig(fig)
            saved_files.append(path)
            visualisations.append((f"Boxplot of {col}", path))

    except Exception as e:
        plt.close("all")
        return f"Error in generate_visuals: {e}"
    
    return visualisations, saved_files

async def cleanup(files):
    for file in files:
        try: os.remove(file)
        except: pass
            


@cl.on_chat_start
async def start():
    await cl.Message(content = "Hi, I'm your EDA agent. I'll help you analyze your data using Gemini.").send()

    files = await cl.AskFileMessage(content = "Please upload your dataset", accept = ["csv", "xlsx", "xls", "txt"]).send()
    
    if not files:
        await cl.Message(content = "No dataset uploaded. Please upload a dataset.").send()
        return
    
    processing_msg = await cl.Message(content = "Processing dataset...").send()
    await processing_msg.update(content = "Analyzing dataset...")
    
    try:
        file = files[0]
        file_name = file.name.lower()
        
        # Check file size (limit to 50MB)
        if len(file.content) > 50 * 1024 * 1024:
            await processing_msg.update(content = "File too large. Please upload a file smaller than 50MB.")
            return
        
        # Handle different file formats
        if file_name.endswith('.csv'):
            try:
                content = file.content.decode("utf-8", errors="replace")
                df = pd.read_csv(io.StringIO(content))
            except Exception as e:
                await processing_msg.update(content = f"Error reading CSV file: {str(e)}")
                return
                
        elif file_name.endswith(('.xlsx', '.xls')):
            try:
                # For Excel files, we need to read from bytes
                df = pd.read_excel(io.BytesIO(file.content))
            except Exception as e:
                await processing_msg.update(content = f"Error reading Excel file: {str(e)}")
                return
                
        elif file_name.endswith('.txt'):
            try:
                content = file.content.decode("utf-8", errors="replace")
                # Try to detect if it's tab-separated or comma-separated
                if '\t' in content[:100]:  # Check first 100 chars for tabs
                    df = pd.read_csv(io.StringIO(content), sep='\t')
                else:
                    df = pd.read_csv(io.StringIO(content))
            except Exception as e:
                await processing_msg.update(content = f"Error reading text file: {str(e)}")
                return
        else:
            await processing_msg.update(content = f"Unsupported file format: {file_name}. Please upload CSV, Excel, or TXT files.")
            return
            
        if df.empty:
            await processing_msg.update(content = "Empty dataset. Please check your file and try again.")
            return
    
        cl.user_session.set("df", df)

        info = df_info_string(df)
        await cl.Message(content = info).send()

        if GEMINI_AVAILABLE:
            plan = await ai_text_analysis("plan", info)
            await cl.Message(content = f" AI Plan: \n{plan}").send()

        visuals, saved_files = generate_visuals(df)
        
        # Check if visuals is an error message
        if isinstance(visuals, str):
            await processing_msg.update(content = f"Error generating visuals: {visuals}")
            return
            
        for title, path in visuals:
            await cl.Message(content = f"**{title}**", elements = [cl.Image(name = title, path = path)]).send()

        if GEMINI_AVAILABLE:
            insights = await ai_vision_analysis(visuals)
            for title, insight in insights:
                await cl.Message(content = f"### {title} Insight \n {insight}").send()

            final = await ai_text_analysis("final", info)
            await cl.Message(content = f"### Final AI Report: \n{final}").send()

        await processing_msg.update(content = "Analysis Completed.")
        await cleanup(saved_files)

    except Exception as e:
        traceback.print_exc()
        await processing_msg.update(content = f"Error: {e}")
