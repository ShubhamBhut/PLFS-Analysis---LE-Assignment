# app.py
import streamlit as st
import plfs_processor  # Your script

st.title("PLFS Data Processor for Non-Coders")

# 1. Drag & Drop Interface
hh_file = st.file_uploader("Upload Household File (.dta)", type="dta")
per_file = st.file_uploader("Upload Person File (.dta)", type="dta")
layout_file = st.file_uploader("Upload Layout File (.xlsx)", type="xlsx")

# 2. The "Magic Button"
if st.button("Generate Clean Dataset"):
    if hh_file and per_file and layout_file:
        st.write("Processing... This may take a minute.")
        
        # Call your main function (modified to accept file objects instead of paths)
        df_final = plfs_processor.process_from_upload(hh_file, per_file, layout_file)
        
        st.success("Done!")
        st.write(df_final.head())
        
        # 3. Download Button
        csv = df_final.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "clean_plfs_data.csv", "text/csv")
    else:
        st.error("Please upload all 3 files first!")