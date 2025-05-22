import sys
import os
import zipfile
import io

# Define the backend path explicitly
backend_path = '/Users/ktrizzo/Documents/code/Git Projects/photorch-app/backend'

# Remove any unwanted paths (like the old src path)
unwanted_path = '/Users/ktrizzo/Documents/code/Git Projects/PhoTorch edits/photorch/src'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

# Add the backend path to sys.path (only if it's not already included)
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Print sys.path for debugging purposes
print("sys.path:", sys.path)



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from backend.fvcb import fitaci
from backend.fvcb import initphotodata
from backend.util import *
plt.rcParams['font.family'] = 'serif'

if not hasattr(st, "_has_printed_startup_message"):
    print("🚀 App running at http://localhost:8501")
    st._has_printed_startup_message = True


# Smooth Max
def smax(a, b, k):
    h = np.maximum(0, 1 - np.abs(a - b) / (4 * k))
    return np.maximum(a, b) + h**2 * k

# Smooth Min
def smin(a, b, k):
    h = np.maximum(0, 1 - np.abs(a - b) / (4 * k))
    return np.minimum(a, b) - h**2 * k

# Pressure Volume Relationship (psi = f(RWC))
def PV(x, pio, Rtlp, elasticity):
    R = x
    pi = pio / R
    P = - pio * smax(0, (R - Rtlp) / (1 - Rtlp), 0.01)**elasticity
    psi = P + pi
    return psi

# Buckley, Turnbull, Adams stomatal conductance model (2012)
def BTA(x, Em, i0, k, b):
    q,d = x
    return Em * (q+i0) / (k + b*q + (q+i0) * d)


# R-squared function
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Streamlit Interface
st.title('PhoTorch')
st.subheader('A tool for fitting common plant physiological models')

# Create tab navigation
tabs = st.tabs(["Photosynthesis", "Stomatal Conductance", "Pressure-Volume","PROSPECT"])


# ---- PHOTOSYNTHESIS MODEL ----
with tabs[0]:
    st.header("Photosynthesis Model Fitting")
    # File uploader
    uploaded_files = st.file_uploader("Upload photosynthesis data files", accept_multiple_files=True)
    uploaded_filenames = [file.name for file in uploaded_files] if uploaded_files else []

    for file in uploaded_files:
        filename = file.name
        name, ext = os.path.splitext(filename)

        # Accept if .txt or extensionless
        if ext == ".txt" or ext == "" or ext ==".xlsx" or ext ==".csv":
            pass
        else:
            st.warning(f"Unsupported file type: {filename}")

    if uploaded_filenames != st.session_state.get("last_uploaded_files", []):
        st.session_state["fit_done"] = False
        st.session_state["last_uploaded_files"] = uploaded_filenames

    if uploaded_files:
        header_present = st.toggle("Skip Header Lines", value=True)
        rescale_with_survey = st.toggle("Rescale with Survey Data",value=True)
        dfs = []
        survey_dfs = []
        for numCurve, file in enumerate(uploaded_files):
            # Set skiprows based on checkbox and file type
            if not header_present:
                skiprows = 0
            elif file.name.endswith(".xlsx"):
                skiprows = 14
            else:
                skiprows = 66

            # Load file with appropriate skiprows
            if file.name.endswith(".xlsx"):
                df = pd.read_excel(file, skiprows=skiprows)
            elif file.name.endswith(".txt"):
                df = pd.read_csv(file, skiprows=skiprows, sep="\t")
            elif file.name.endswith(".csv"):
                df = pd.read_csv(file, skiprows=skiprows)
            else:
                df = pd.read_csv(file, skiprows=skiprows, sep="\t")

            # Drop the first row if it exactly matches the column names
            if header_present:
                df = df.drop(index=0).reset_index(drop=True)

            # Route file to survey or main list
            if "survey" in file.name.lower():
                survey_dfs.append(df)
            else:
                df["CurveID"] = numCurve
                dfs.append(df)

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        survey_df = pd.concat(survey_dfs, ignore_index=True) if survey_dfs else pd.DataFrame()

        st.success(f"✅ Loaded {len(df)} rows from {len(uploaded_files)} files.")
        num_survey = len(survey_dfs)
        num_curves = len(dfs)

        st.info(f"📊 Loaded {num_curves} response curve file(s) and {num_survey} survey file(s).")
        st.write(f"Response Curve Data")
        st.dataframe(df.head(),hide_index=True)

        st.write(f"Survey Data")
        if survey_df.empty:
            st.info(f"None")
        else:
            st.dataframe(survey_df.head(),hide_index=True)

        # Rescale if toggle is on
        if rescale_with_survey and not survey_df.empty and not df.empty:
            try:
                survey_A_median = pd.to_numeric(survey_df["A"], errors="coerce").median()

                # Convert Q and Ci to numeric to filter
                df["Qabs"] = pd.to_numeric(df["Qabs"], errors="coerce")
                df["Ci"] = pd.to_numeric(df["Ci"], errors="coerce")
                df["Tleaf"] = pd.to_numeric(df["Tleaf"], errors="coerce")
                df["A"] = pd.to_numeric(df["A"], errors="coerce")

                target_mask = (df["Qabs"].between(0.85*1900, 0.85*2100)) & (df["Tleaf"].between(24, 26))
                max_A_at_target = df.loc[target_mask, "A"].nlargest(10).median()
                #st.write(max_A_at_target)

                if pd.notna(max_A_at_target) and pd.notna(survey_A_median) > 0:
                    scale_factor = survey_A_median / max_A_at_target
                    df["A"] = df["A"] * scale_factor
                    st.success(f"Rescaling performed. Survey median: {survey_A_median:.2f}, response curve target: {max_A_at_target:.2f}.")
                elif not pd.notna(max_A_at_target):
                    st.info(f"Rescaling ignored. Response curve target conditions don't exist.")
            except Exception as e:
                st.warning(f"Rescaling failed: {e}")

        # Try auto-detecting Q, T, Ci, A
        default_cols = {
            "Qabs": "Qabs",
            "Tleaf": "Tleaf",
            "Ci": "Ci",
            "A": "A"
        }

        found_cols = {}
        for key, colname in default_cols.items():
            found_cols[key] = colname if colname in df.columns else None

        # Check if all required columns were found
        if all(found_cols.values()):
            st.subheader("Auto-Detected Columns For Fitting")
            #st.write({k: found_cols[k] for k in ["Q", "T", "Ci", "A"]})
            selected_data = df[[found_cols["Qabs"], found_cols["Tleaf"], found_cols["Ci"], found_cols["A"],"CurveID"]].copy()
            selected_data.columns = ["Qabs", "Tleaf", "Ci", "A","CurveID"]
            st.dataframe(selected_data.head(),hide_index=True)

            # Store in session state for downstream use
            st.session_state["selected_data"] = selected_data

            # Initialize reselect flag if not already present
            if "reselect_columns" not in st.session_state:
                st.session_state["reselect_columns"] = False

            if st.button("Reselect Columns"):
                st.session_state["reselect_columns"] = True

        else:
            st.warning("Could not auto-detect all required columns. Please select manually.")
            st.session_state["reselect_columns"] = True

        # ---- MANUAL COLUMN SELECTION ----
        if st.session_state.get("reselect_columns", True):
            st.write("### Select Model Columns")

            col_options = list(df.columns)

            q_col = st.selectbox("Qabs (Light)", col_options, index=col_options.index(found_cols["Qabs"]) if found_cols["Qabs"] else 0)
            t_col = st.selectbox("Tleaf (Temperature)", col_options, index=col_options.index(found_cols["Tleaf"]) if found_cols["Tleaf"] else 1)
            ci_col = st.selectbox("Ci (Internal CO₂)", col_options, index=col_options.index(found_cols["Ci"]) if found_cols["Ci"] else 2)
            a_col = st.selectbox("A (Assimilation)", col_options, index=col_options.index(found_cols["A"]) if found_cols["A"] else 3)

            # Auto-update preview on every selection change
            selected_data = df[[q_col, t_col, ci_col, a_col, "CurveID"]].copy()
            selected_data.columns = ["Qabs", "Tleaf", "Ci", "A","CurveID"]
            st.dataframe(selected_data.head(), hide_index=True)

            # Update the session state temporarily
            st.session_state["selected_data"] = selected_data
            


        df = st.session_state.get("selected_data")
        st.session_state["selected_data"] = df

        required_cols = {"Qabs", "Tleaf", "Ci", "A","CurveID"}
        if not required_cols.issubset(df.columns):
            st.error(f"Selected data is missing one or more required columns: {required_cols - set(df.columns)}")
            st.stop()
        
        #st.write("Model fitting process will be implemented here.")
        species_to_fit = st.text_input("Enter species name", "Species")
        species_variety = st.text_input("Enter species variety", "Variety")

        # User Inputs for fitting settings
        LightResponseType = st.selectbox("Select Light Response Type", [1, 2], index=1)
        TemperatureResponseType = st.selectbox("Select Temperature Response Type", [1, 2], index=1)
        Fitgm = st.checkbox("Fit gm (Mesophyll conductance)", value=False)
        FitGamma = st.checkbox("Fit Gamma (Photorespiration)", value=False)
        FitKc = st.checkbox("Fit Kc (Carboxylation)", value=False)
        FitKo = st.checkbox("Fit Ko (Oxygenation)", value=False)
        
        # Advanced Hyperparameters Section
        learningRate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.08)
        iterations = st.slider("Iterations", min_value=1000, max_value=10000, value=1500)

        for col in ["Qabs", "Tleaf", "Ci", "A","CurveID"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna().reset_index(drop=True)

        if st.button("Fit"):
            st.session_state["fit_done"] = True
            lcd = initphotodata.initLicordata(df, preprocess=True)
            device_fit = 'cpu'
            lcd.todevice(torch.device(device_fit))
            
            fvcb = fitaci.initM.FvCB(
                lcd,
                LightResp_type=LightResponseType,
                TempResp_type=TemperatureResponseType,
                onefit=True,
                fitgamma=FitGamma,
                fitKc=FitKc,
                fitKo=FitKo,
                fitgm=Fitgm
            )
            with st.spinner(""):
                progress_text = st.empty()
                fitresult = fitaci.run(
                    fvcb,
                    learn_rate=learningRate,
                    maxiteration=iterations,
                    minloss=1,
                    recordweightsTF=False,
                    progress_callback = progress_text
                )
                fvcb = fitresult.model

            # Display parameters
            printFvCBParameters(fvcb, LightResponseType, TemperatureResponseType, Fitgm, FitGamma, FitKc, FitKo)
            param_dict = {
                "Vcmax25": fvcb.Vcmax25.item(),
                "Jmax25": fvcb.Jmax25.item(),
            }

            if TemperatureResponseType == 2:
                param_dict["Vcmax_Topt"] = fvcb.TempResponse.Topt_Vcmax.item()
                param_dict["Jmax_Topt"] = fvcb.TempResponse.Topt_Jmax.item()

            param_dict["alpha"] = fvcb.LightResponse.alpha.item()

            if LightResponseType == 2:
                param_dict["theta"] = fvcb.LightResponse.theta.item()
            if Fitgm:
                param_dict["gm"] = fvcb.gm.item()
            if FitGamma:
                param_dict["Gamma25"] = fvcb.Gamma25.item()
            if FitKc:
                param_dict["Kc"] = fvcb.Kc25.item()
            if FitKo:
                param_dict["Ko"] = fvcb.Ko25.item()

            df = pd.DataFrame(param_dict.items(), columns=["Parameter", "Value"])
            filename = f"{species_to_fit}_{species_variety}_FvCB_Parameters.csv"
            savepath = os.path.join("results", "parameters", filename)

        
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            vars = ["species", "variety", "Vcmax25", "Jmax25", "TPU25", "Rd25", "alpha", "theta", "Vcmax_dHa", "Vcmax_Topt", "Vcmax_dHd",
                    "Jmax_dHa", "Jmax_Topt", "Jmax_dHd", "TPU_dHa", "TPU_Topt", "TPU_dHd", "Rd_dHa", "Gamma25", "Gamma_dHa",
                    "Kc25", "Kc_dHa", "Ko25", "Ko_dHa", "O"]
            
            # Helper for placeholder value
            def t2(x_name):
                return getattr(fvcb.TempResponse, x_name).item() if TemperatureResponseType == 2 and hasattr(fvcb.TempResponse, x_name) else 99999

            def t1(x_name):
                return getattr(fvcb.TempResponse, x_name).item() if TemperatureResponseType == 2 and hasattr(fvcb.TempResponse, x_name) else 1

            theta = fvcb.LightResponse.theta.item() if LightResponseType == 2 else 0.0

            vals = [
                species_to_fit,
                species_variety,
                fvcb.Vcmax25.item(),
                fvcb.Jmax25.item(),
                fvcb.TPU25.item(),
                fvcb.Rd25.item(),
                fvcb.LightResponse.alpha.item(),
                theta,
                fvcb.TempResponse.dHa_Vcmax.item() if hasattr(fvcb.TempResponse, "dHa_Vcmax") else 99999,
                t2("Topt_Vcmax"),
                t1("dHd_Vcmax"),
                fvcb.TempResponse.dHa_Jmax.item() if hasattr(fvcb.TempResponse, "dHa_Jmax") else 99999,
                t2("Topt_Jmax"),
                t1("dHd_Jmax"),
                fvcb.TempResponse.dHa_TPU.item() if hasattr(fvcb.TempResponse, "dHa_TPU") else 99999,
                t2("Topt_TPU"),
                t1("dHd_TPU"),
                fvcb.TempResponse.dHa_Rd.item() if hasattr(fvcb.TempResponse, "dHa_Rd") else 99999,
                fvcb.Gamma25.item(),
                fvcb.TempResponse.dHa_Gamma.item() if hasattr(fvcb.TempResponse, "dHa_Gamma") else 99999,
                fvcb.Kc25.item(),
                fvcb.TempResponse.dHa_Kc.item() if hasattr(fvcb.TempResponse, "dHa_Kc") else 99999,
                fvcb.Ko25.item(),
                fvcb.TempResponse.dHa_Ko.item() if hasattr(fvcb.TempResponse, "dHa_Ko") else 99999,
                fvcb.Oxy.item()
            ]

            params = pd.DataFrame([vals], columns=vars)
            params.to_csv(savepath, index=False)

            with open(savepath, "rb") as f:
                file_bytes = f.read()

            st.session_state["last_param_table"] = df
            st.session_state["last_param_dict"] = params
            st.session_state["last_filename"] = filename
            st.session_state["last_file_bytes"] = file_bytes


        if st.session_state.get("fit_done", False):
            st.success(f"✅ Parameters saved as: `{st.session_state['last_filename']}`")
            st.dataframe(st.session_state["last_param_dict"],hide_index=True)
            st.download_button(
                "Download Parameters CSV",
                st.session_state["last_file_bytes"],
                file_name=st.session_state["last_filename"],
                mime="text/csv"
            )
            
            df = st.session_state["selected_data"]
            Q_meas = df["Qabs"].values
            Ci_meas = df["Ci"].values
            Tleaf_meas = df["Tleaf"].values + 273.15  # convert to K if needed
            A_meas = df["A"].values

            p = st.session_state["last_param_dict"];



            # ---------------- LIGHT RESPONSE ----------------
            Q = np.linspace(0, 2000, 60)
            Ci = np.full_like(Q, 300)
            T = np.full_like(Q, 25 + 273.15)

            x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
            A = evaluateFvCB(x, p)

            fig_Q, ax = plt.subplots(figsize=(10, 10))
            ax.plot(Q, A, "r", linewidth=4, label="FvCB Fit at Ci=300, T=25")

            filtered_df = df[
                (df["Ci"] >= 290) & (df["Ci"] <= 310) &
                (df["Tleaf"] >= 24.5) & (df["Tleaf"] <= 25.5)
            ]
            ax.scatter(df["Qabs"], df["A"], c="gainsboro", s=25, label="All Measured A")
            ax.scatter(filtered_df["Qabs"], filtered_df["A"], c="k", s=25, label="Relevant Measured A")

            ax.set_xlabel(r"Q (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax.set_ylabel(r"A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            ax.set_ylim([0, max(1, max(A_meas) * 1.1)])
            ax.set_xticks([0, 1000, 2000])
            ax.legend(fontsize=16)
            st.pyplot(fig_Q)

            # ------------- Ci RESPONSE -------------
            Ci_range = np.linspace(0, 2000, 100)
            Q_fixed = np.full_like(Ci_range, 0.85*2000)
            T_fixed = np.full_like(Ci_range, 25 + 273.15)

            x_Ci = np.column_stack((Ci_range, Q_fixed, T_fixed))
            A_Ci = evaluateFvCB(x_Ci, p)

            fig_ci, ax_ci = plt.subplots(figsize=(10, 10))
            ax_ci.plot(Ci_range, A_Ci, "r", linewidth=4, label="FvCB Fit at Q=2000, T=25")

            filtered_df = df[
                (df["Qabs"] >= 0.85*1900) & (df["Qabs"] <= 0.85*2100) &
                (df["Tleaf"] >= 23.5) & (df["Tleaf"] <= 26)
            ]
            ax_ci.scatter(df["Ci"], df["A"], c="gainsboro", s=25, label="All Measured A")
            ax_ci.scatter(filtered_df["Ci"], filtered_df["A"], c="k", s=25, label="Relevant Measured A")

            ax_ci.set_xlabel(r"C$_i$ (µmol mol$^{-1}$)", fontsize=16)
            ax_ci.set_ylabel(r"A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax_ci.tick_params(axis='both', labelsize=14)
            for spine in ax_ci.spines.values():
                spine.set_linewidth(2)
            ax_ci.set_ylim([0, max(1, max(A_meas) * 1.1)])
            ax_ci.legend(fontsize=16)
            st.pyplot(fig_ci)

            # ------------- T RESPONSE -------------
            T_range = np.linspace(20, 50, 100) + 273.15
            Ci_fixed = np.full_like(T_range, 300)
            Q_fixed = np.full_like(T_range, 0.85*2000)

            x_T = np.column_stack((Ci_fixed, Q_fixed, T_range))
            A_T = evaluateFvCB(x_T, p)

            fig_T, ax_T = plt.subplots(figsize=(10, 10))
            ax_T.plot(T_range - 273.15, A_T, "r", linewidth=4, label="FvCB Fit at Ci=300, Q=2000")

            filtered_df = df[
                (df["Qabs"] >= 0.85*1900) & (df["Qabs"] <= 0.85*2100) &
                (df["Ci"] >= 290) & (df["Ci"] <= 310)
            ]
            ax_T.scatter(df["Tleaf"], df["A"], c="gainsboro", s=25, label="All Measured A")
            ax_T.scatter(filtered_df["Tleaf"], filtered_df["A"], c="k", s=25, label="Relevant Measured A")

            ax_T.set_xlabel(r"T (°C)", fontsize=16)
            ax_T.set_ylabel(r"A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax_T.tick_params(axis='both', labelsize=14)
            for spine in ax_T.spines.values():
                spine.set_linewidth(2)
            ax_T.set_ylim([0, max(1, max(A_meas) * 1.1)])
            ax_T.legend(fontsize=16)
            st.pyplot(fig_T)

            x_all = np.column_stack((
                df["Ci"].values,
                df["Qabs"].values,
                df["Tleaf"].values + 273.15  # convert °C to K
            ))

            # ------------- 1:1 -------------
            A_model = evaluateFvCB(x_all, p)
            A_measured = df["A"].values

            A_measured = np.array(A_measured)
            A_model = np.array(A_model)
            valid = ~np.isnan(A_measured) & ~np.isnan(A_model)
            A_model = A_model[valid]
            A_measured = A_measured[valid]

            fig_1to1, ax_1to1 = plt.subplots(figsize=(8, 8))
            ax_1to1.scatter(A_measured, A_model, color='k', s=10, label="")

            lims = [min(min(A_measured), min(A_model)), max(max(A_measured), max(A_model))]
            ax_1to1.plot(lims, lims, 'k--', linewidth=2, label="1:1")

            ax_1to1.set_xlabel("Measured A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax_1to1.set_ylabel("Modeled A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax_1to1.set_xlim(lims)
            ax_1to1.set_ylim(lims)
            ax_1to1.tick_params(axis='both', labelsize=14)
            for spine in ax_1to1.spines.values():
                spine.set_linewidth(2)
            ax_1to1.legend(fontsize=14)
            ax_1to1.set_aspect('equal', 'box')

            r2 = computeR2(A_measured, A_model)
            ax_1to1.text(0.05, 0.95, f"R$^2$ = {r2:.2f}", transform=ax_1to1.transAxes,
                        fontsize=14, verticalalignment='top',fontfamily="serif")

            st.pyplot(fig_1to1)

            # Create a grid for Ci and T
            Ci = np.linspace(100, 2000, 60)        
            T = np.linspace(273, 50 + 273, 60)        
            Ci, T = np.meshgrid(Ci, T)          
            Q = 2000 * np.ones_like(T)                

            # First subplot: A vs Ci and T at Q = 2000
            fig3D = plt.figure(figsize=(10, 10))
            ax1 = fig3D.add_subplot(1, 1, 1, projection='3d')
            
            x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
            A = evaluateFvCB(x, p)  # Run the FvCB model
            A = A.reshape(Ci.shape)  # Reshape to match the grid

            # Plot modeled surface
            ax1.plot_surface(Ci, T - 273.15, A, edgecolor='none', alpha=0.5, label="FvCB Fit")
            ax1.set_xlabel(r"C$_i$ (µmol mol$^{-1}$)", fontsize=16, labelpad=15)
            ax1.set_ylabel(r"T ($^{\circ}$C)", fontsize=16, labelpad=15)
            ax1.set_zlabel(r"A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
            ax1.view_init(elev=5, azim=-10)
            ax1.tick_params(axis='both', labelsize=10)
            for spine in ax1.spines.values():
                spine.set_linewidth(4)
            ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            # Plot measured data
            ax1.scatter(df["Ci"][df["Qabs"]>0.85*1900], df["Tleaf"][df["Qabs"]>0.85*1900], df["A"][df["Qabs"]>0.85*1900], c='r', s=30, label="A-Ci Curves")
            ax1.set_xticks([0,1000,2000])
            ax1.legend(loc="upper right",fontsize=16)
            st.pyplot(fig3D)


            # Second subplot: A vs Ci and Q at T = 298.15 K
            fig3D2 = plt.figure(figsize=(10, 10))
            ax2 = fig3D2.add_subplot(1, 1, 1, projection='3d')
            Ci = np.linspace(5, 2000, 60)
            Q = np.linspace(0, 2000, 60)
            Ci, Q = np.meshgrid(Ci, Q)
            T = 298.15 * np.ones_like(Ci)  # Constant temperature at 298.15 K

            x = np.column_stack((Ci.ravel(), Q.ravel(), T.ravel()))
            A = evaluateFvCB(x, p)
            A = A.reshape(Ci.shape)

            # Plot modeled surface
            #ax2.plot_surface(Ci, Q, A, cmap='YlGn', edgecolor='none', alpha=0.8,label="FvCB Fit")
            ax2.plot_surface(Ci, Q, A, alpha=0.5,label="FvCB Fit",linewidth=8)
            ax2.set_xlabel(r"C$_i$ ($\mu$mol mol$^{-1}$)", fontsize=18, labelpad=15)
            ax2.set_ylabel(r"Q ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=18, labelpad=15)
            ax2.set_zlabel(r"A ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=18, labelpad=5)
            ax2.tick_params(axis='both', labelsize=13)
            for spine in ax2.spines.values():
                spine.set_linewidth(4)
            ax2.view_init(elev=5, azim=-10)
            ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


            # Plot measured data on modeled surface
            ax2.scatter(df["Ci"][df["Tleaf"]<27], df["Qabs"][df["Tleaf"]<27], df["A"][df["Tleaf"]<27], c='r', s=30,label="A-Ci Curves")
            ax2.set_xticks([0,1000,2000])
            ax2.legend(loc="upper right",fontsize=16)

            plt.tight_layout()
            st.pyplot(fig3D2)

            if(num_survey>0):
                # ------------- Survey -------------
                for col in ["obs", "Tleaf", "Ci", "A"]:
                    survey_df[col] = pd.to_numeric(survey_df[col], errors='coerce')

                survey_df = survey_df.dropna(subset=["obs", "A"])
                survey_df_sorted = survey_df.sort_values(by="A", ascending=True).reset_index(drop=True)
                median_A = survey_df_sorted["A"].median()
                std_A = survey_df_sorted["A"].std()
                figSurvey, ax = plt.subplots(figsize=(10, 10))

                ax.bar(range(len(survey_df_sorted)), survey_df_sorted["A"], color="k")
                ax.axhline(median_A, color="red", linestyle="--", linewidth=4, label=f"Median A = {median_A:.2f}")
                ax.axhline(median_A, color="red", linestyle="--", linewidth=4, label=f"Std A = {std_A:.2f}")


                ax.set_xlabel("Observation", fontsize=16)
                ax.set_ylabel(r"A (µmol m$^{-2}$ s$^{-1}$)", fontsize=16)
                ax.set_title("Survey Measurements")
                ax.tick_params(axis='both', labelsize=14)
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                ax.legend(fontsize=16)
                plt.tight_layout()
                st.pyplot(figSurvey)

            # Create in-memory zip archive
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                # Add each figure in both .png and .eps formats
                figures = []
                if 'fig_Q' in locals() and fig_Q is not None:
                    figures.append(("light_response", fig_Q))
                if 'fig_ci' in locals() and fig_ci is not None:
                    figures.append(("aci_response", fig_ci))
                if 'fig_T' in locals() and fig_T is not None:
                    figures.append(("temp_response", fig_T))
                if 'fig_1to1' in locals() and fig_1to1 is not None:
                    figures.append(("one_to_one", fig_1to1))
                if 'fig3D' in locals() and fig3D is not None:
                    figures.append(("aci_temp_surface", fig3D))
                if 'fig3D2' in locals() and fig3D2 is not None:
                    figures.append(("aci_light_surface", fig3D2))
                if 'figSurvey' in locals() and figSurvey is not None:
                    figures.append(("survey", figSurvey))

                for name, figure in figures:
                    # Save as PNG
                    png_buf = io.BytesIO()
                    figure.savefig(png_buf, format='png', bbox_inches='tight')
                    png_buf.seek(0)
                    zip_file.writestr(f"{name}.png", png_buf.read())

                    # Save as EPS
                    eps_buf = io.BytesIO()
                    figure.savefig(eps_buf, format='eps', bbox_inches='tight')
                    eps_buf.seek(0)
                    zip_file.writestr(f"{name}.eps", eps_buf.read())

            # Finalize zip
            zip_buffer.seek(0)

            # Download button
            st.download_button(
                label="📥 Download Figures (.eps + .png)",
                data=zip_buffer,
                file_name="FvCB_Figures.zip",
                mime="application/zip"
            )


# ---- STOMATAL CONDUCTANCE ----
with tabs[1]:

    st.header("Stomatal Conductance Model Fitting")
    uploaded_file = st.file_uploader("Upload stomatal conductance data file", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
        st.dataframe(df.head(), hide_index=True)
        species_col = "species" if "species" in df.columns else None

        # ---- MODEL SELECTION ----
        model_name = st.selectbox("Select Stomatal Conductance Model", ["Buckley, Turnbull, Adams (2012)", "Medlyn et al. (2011)",  "Leuning et al. (1995)", "Ball, Woodrow, Berry (1987)"])
        if(model_name=="Buckley, Turnbull, Adams (2012)"):
            st.latex(r"g_{sw}(Q,D) = \frac{E_m (Q+i_0)}{k+bQ+(Q+i_0)D}")

        if(model_name=="Medlyn et al. (2011)"):
            st.latex(r"g_{sw}(A,D,C_a) = g_0 + \left(1+\frac{g_1}{\sqrt{D}}\right)\frac{A}{C_a}")

        if(model_name=="Leuning et al. (1995)"):
            st.latex(r"g_{sw}(A,C_s,\Gamma) = g_0 + \frac{g_1 A}{(C_s - \Gamma)(1+\frac{D}{D_0})}")
        
        if(model_name=="Ball, Woodrow, Berry (1987)"):
            st.latex(r"g_{sw}(A,H,C_s) = g_0 + g_1 A \frac{H}{C_s}")
        # ---- MODEL-SPECIFIC REQUIRED COLUMNS ----
        model_required_cols = {
            "Buckley, Turnbull, Adams (2012)": ["gsw", "Qabs", "D"],
            "Medlyn et al. (2011)": ["gsw", "A", "D", "Ca"],
            "Leuning et al. (1995)" : ["gsw", "A", "D"],
            "Ball, Woodrow, Berry (1987)": ["gsw", "A", "Cs", "H"]
        }

        # ---- MODEL-SPECIFIC COLUMN OPTIONS FOR AUTO-DETECT ----
        model_column_options = {
            "gsw": ["g", "gs", "gsw"],
            "A": ["A", "An", "Anet", "Assimilation"],
            "Cs": ["Cs","cs","Ca","ca"],
            "Ca": ["Ca", "ca"],
            "H": ["RH", "rh","H","h","rh_s","RHsam","RH_sam"],
            "Tleaf": ["Tleaf", "T", "Temp"],
            "D": ["VPDleaf","VPD", "vpd","D"],
            "Qamb": ["PPFD", "Q", "Qin", "Qamb","PAR"],
            "Qabs": ["PPFD", "Q", "Qin", "Qamb","PAR"],
            "Gamma" : ["Gamma","gamma","compensation_point"]
        }

        required_keys = model_required_cols[model_name]
        found_cols = {}

        # ---- AUTO-DETECTION ----
        for key in required_keys:
            options = model_column_options.get(key, [])
            found = next((col for col in df.columns if col in options), None)
            found_cols[key] = found

        if all(found_cols.values()):
            st.subheader("Auto-Detected Columns For Fitting")
            selected_data = df[[found_cols[k] for k in required_keys]].copy()
            selected_data.columns = required_keys
            st.dataframe(selected_data.head(), hide_index=True)

            st.session_state["selected_data"] = selected_data

            st.session_state["reselect_gs_columns"] = False

            if st.button("Reselect Columns Manually"):
                st.session_state["reselect_gs_columns"] = True
        else:
            st.warning("Could not auto-detect all required columns for this model. Please select manually.")
            st.session_state["reselect_gs_columns"] = True

        # ---- MANUAL COLUMN SELECTION ----
        if st.session_state.get("reselect_gs_columns", False):
            st.write(f"### Select Columns for {model_name}")

            col_options = list(df.columns)
            manual_selection = {}

            for key in required_keys:
                default_index = col_options.index(found_cols[key]) if found_cols[key] in col_options else 0
                manual_selection[key] = st.selectbox(f"{key}", col_options, index=default_index)

            selected_data = df[[manual_selection[k] for k in required_keys]].copy()
            selected_data.columns = required_keys
            st.dataframe(selected_data.head(), hide_index=True)

            st.session_state["selected_data"] = selected_data
        
        # --- UNIT CONVERSIONS ---
        st.markdown("---")

        abs_PAR = 0.85

        gsw_candidates = ["g", "gs", "gsw"]
        gsw_col = next((col for col in selected_data.columns if col in gsw_candidates), None)

        # Light-related unit conversions
        light_candidates = ["PPFD", "Q", "Qin", "Qamb", "PAR", "Qabs"]
        light_col = next((col for col in selected_data.columns if col in light_candidates), None)

        if light_col:
            st.write("### Select Q (Light) Units")
            q_unit = st.radio(
                "What unit is the given light flux data in?",
                ["μmol/m²/s ambient", "μmol/m²/s absorbed", "W/m² ambient", "W/m² absorbed"],
                horizontal=True,
                key="q_unit_radio"
            )

            if q_unit == "μmol/m²/s ambient":
                selected_data[light_col] *= abs_PAR
            elif q_unit == "W/m² ambient":
                selected_data[light_col] *= abs_PAR * 4.57
            elif q_unit == "W/m² absorbed":
                selected_data[light_col] *= 4.57
            # else: μmol/m²/s absorbed, do nothing

        # VPD-related unit conversions
        vpd_candidates = ["VPD", "vpd", "VPDleaf", "D"]
        vpd_col = next((col for col in selected_data.columns if col in vpd_candidates), None)

        if vpd_col:
            st.write("### Select D (VPD) Units")
            d_unit = st.radio(
                "What unit is the given leaf VPD data in?",
                ["kPa", "mol/mol", "mmol/mol"],
                horizontal=True,
                key="d_unit_radio"
            )

            if d_unit == "kPa":
                selected_data[vpd_col] *= 1000.0 / 101.3  # Convert kPa to mmol/mol
            elif d_unit == "mol/mol":
                selected_data[vpd_col] *= 1000.0
            # else: mmol/mol, no change

        # --- ADVANCED OPTIONS ---
        with st.expander("Advanced Options"):
            st.write("Customize additional settings for model fitting.")

            # set_PAR_abs = st.checkbox("Set leaf PAR absorptivity (default = 0.85)", key="custom_par_check")
            # if set_PAR_abs:
            #     new_abs_PAR = st.number_input("Leaf PAR absorptivity", min_value=0.1, max_value=1.0, value=0.85, step=0.01)
            #     selected_data[light_col] *= new_abs_PAR / abs_PAR 
            #     abs_PAR = new_abs_PAR 

            if species_col:
                fit_all_species = st.checkbox("Fit each species in file?", key="fit_all_sc")

        if st.button("Fit Model",key = "fit_sc"):
            if model_name == "Buckley, Turnbull, Adams (2012)":
                x = selected_data[light_col]
                y = selected_data[vpd_col]
                z = selected_data[gsw_col]

                # if species_col:
                #     if fit_all_species:
                #         st.latex(r"g_{sw}(Q,D) = \frac{E_m (Q+i_0)}{k+bQ+(Q+i_0)D}")

                #         species_fits = []
                #         fig, ax = plt.subplots()

                #         for species in species_list:
                #             data_sc = all_data_sc
                #             species_data_sc = data_sc[data_sc[species_col] == species]
                #             x = species_data_sc[st.session_state.x_colum_sc]
                #             y = species_data_sc[st.session_state.y_colum_sc]

                #             try:
                #                 popt, _ = curve_fit(lambda X, Em, i0, k, b: BTA(X, Em, i0, k, b), 
                #                         (x, y), z, p0=p0, bounds=bounds)

                #                 Em, i0, k, b = popt


                #                 # Compute R2 and RMSE
                #                 z_pred = BTA((x,y), *popt)
                #                 r2 = r_squared(z, z_pred)
                #                 rmse_val = rmse(z, z_pred)

                #                 # Store results
                #                 species_fits.append({
                #                     'species': species,
                #                     'E_m,': round(Em, 4),
                #                     'i_0': round(i0, 4),
                #                     'k': round(k, 4),
                #                     'b': round(b,4),
                #                     'Rsquared': round(r2, 3),
                #                     'RMSE': round(rmse_val, 3)
                #                 })

                #                 # Plot data and fit
                #                 scatter = ax.plot(x, y, 'o')
                #                 xx = np.linspace(0.5, 1, 100)
                #                 ax.plot(xx, PV(xx, *popt), '-', label=f"{species}",color=scatter[0].get_color())

                #             except Exception as e:
                #                 st.write(f"Error fitting {species}: {e}")

                #         # Display Table
                #         results_df = pd.DataFrame(species_fits)
                #         st.subheader("Best Fit Parameters for All Species")
                #         st.dataframe(results_df)

                #         # Download Button
                #         csv = results_df.to_csv(index=False)
                #         st.download_button("Download Results as CSV", data=csv, file_name="fitted_parameters.csv", mime="text/csv")

                #         # Finalize Plot
                #         ax.set_xlabel("Relative Water Content (/)")
                #         ax.set_ylabel("Leaf Water Potential (MPa)")
                #         ax.legend()
                #         st.pyplot(fig)
                # else:
                st.latex(r"g_{sw}(Q,D) = \frac{E_m (Q+i_0)}{k+bQ+(Q+i_0)D}")
                st.subheader("Best Fit Parameters")
                p0 = [5, 10, 5e3, 5]
                bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])
                print(x)
                try:
                    popt, _ = curve_fit(lambda X, Em, i0, k, b: BTA(X, Em, i0, k, b), 
                                    (x, y), z, p0=p0, bounds=bounds)

                    Em, i0, k, b = popt


                    # Compute R2 and RMSE
                    z_pred = BTA((x,y), Em, i0, k, b)
                    r2 = r_squared(z, z_pred)
                    rmse_val = rmse(z, z_pred)

                    # Display fitted parameters and metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.latex(r"E_m = " + f"{Em:.2f}")  
                    with col2: st.latex(r"i_0 = " + f"{i0:.2f}")
                    with col3: st.latex(r"k = " + f"{k:.2f}")
                    with col4: st.latex(r"b = " + f"{b:.2f}")
                    
                    st.latex(r"R^{2} = " + f"{r2:.3f}")
                    st.latex(r"RMSE = " + f"{rmse_val:.3f}")

                    # Prepare downloadable results
                    if(species_col):
                        results_df = pd.DataFrame({
                            #'species': [selected_species],
                            'Em': [round(Em,4)],
                            'i0': [round(i0,4)],
                            'k': [round(k,4)],
                            'b': [round(b,4)],
                            'Rsquared': [round(r2,4)],
                            'RMSE': [round(rmse_val,4)]
                        })
                    else:
                        results_df = pd.DataFrame({
                            'Em': [round(Em,4)],
                            'i0': [round(i0,4)],
                            'k': [round(k,4)],
                            'b': [round(b,4)],
                            'Rsquared': [round(r2,4)],
                            'RMSE': [round(rmse_val,4)]
                        })
                    csv = results_df.to_csv(index=False)
                    st.dataframe(results_df,hide_index=True)
                    st.download_button(label="Download Results as CSV", data=csv, file_name="BTA_parameters.csv", mime="text/csv")
                    

                    # Plot results
                    # fig, ax = plt.subplots()
                    # ax.plot(x, y, 'ko', label="Measured")
                    # xx = np.linspace(0.5, 1, 100)
                    # ax.plot(xx, PV(xx, *popt), 'r-', label="Modeled")
                    # ax.set_xlabel("Relative Water Content (/)")
                    # ax.set_ylabel("Leaf Water Potential (MPa)")
                    # if species_col:
                    #     ax.set_title(selected_species)
                    # ax.legend()
                    # st.pyplot(fig)

                    # Plot Light Resp
                    Q_meas = x
                    D_meas = y
                    gsw_meas = z

                    Q = np.linspace(0,2000,50)    
                    D = np.linspace(5, 5, 50)

                    fig, ax = plt.subplots(figsize=(10, 10))

                    gsw_modeled = BTA((Q,D), Em,i0,k,b)


                    ax.plot(Q, gsw_modeled,linewidth=4,color="r",label="BMF Fit")
                    ax.set_xlabel(r"$Q$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=16)
                    ax.set_ylabel(r"g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=16)
                    ax.tick_params(axis='both', labelsize=14) 
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)


                    ax.set_ylim([0,max(1,max(max(gsw_meas),max(gsw_modeled))*1.1)])

                    ax.scatter(Q_meas, gsw_meas, c="k", s=25, label="Measured gsw")
                    ax.set_xticks([0,1000,2000])
                    st.pyplot(fig)

                    # Plot VPD Resp

                    Q_meas = x
                    D_meas = y
                    gsw_meas = z

                    Q = np.linspace(2000,2000,50)    
                    D = np.linspace(1, 55, 50)

                    fig, ax = plt.subplots(figsize=(10, 10))

                    gsw_modeled = BTA((Q,D), Em,i0,k,b)


                    ax.plot(D, gsw_modeled,linewidth=4,color="r")
                    ax.set_xlabel(r"$D$ (mmol mol$^{-1}$)", fontsize=16)
                    ax.set_ylabel(r"g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=16)
                    ax.tick_params(axis='both', labelsize=14) 
                    for spine in ax.spines.values():
                        spine.set_linewidth(2)


                    ax.set_ylim([0,max(1,max(max(gsw_meas),max(gsw_modeled))*1.1)])
                    ax.set_xlim([5,60])


                    ax.scatter(D_meas, gsw_meas, c="k", s=25)
                    st.pyplot(fig)


                    # Plot 1:1 Modeled-Measured

                    gsw_pred = BTA((Q_meas,D_meas), Em,i0,k,b)

                    fig_1to1, ax_1to1 = plt.subplots(figsize=(8, 8))
                    ax_1to1.scatter(gsw_meas, gsw_pred, color='k', s=10, label="")

                    lims = [min(min(gsw_meas), min(gsw_pred)), max(max(gsw_meas), max(gsw_pred))]
                    ax_1to1.plot(lims, lims, 'k--', linewidth=2, label="1:1")

                    ax_1to1.set_xlabel("Measured g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=16)
                    ax_1to1.set_ylabel("Modeled g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=16)
                    ax_1to1.set_xlim(lims)
                    ax_1to1.set_ylim(lims)
                    ax_1to1.tick_params(axis='both', labelsize=14)
                    for spine in ax_1to1.spines.values():
                        spine.set_linewidth(2)
                    ax_1to1.legend(fontsize=14)
                    ax_1to1.set_aspect('equal', 'box')

                    r2 = computeR2(gsw_meas, gsw_pred)
                    ax_1to1.text(0.05, 0.95, f"R$^2$ = {r2:.2f}", transform=ax_1to1.transAxes,
                                fontsize=14, verticalalignment='top',fontfamily="serif")

                    st.pyplot(fig_1to1)

                    # Plot 3D Surface

                    Q = np.linspace(0,2000,50)    
                    D = np.linspace(5, 40, 50)   
                    Q,D = np.meshgrid(Q, D)

                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(1, 1, 1, projection='3d')
                    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                    gsw_modeled = BTA((Q,D), Em,i0,k,b)

                    ax.scatter(Q_meas, D_meas, gsw_meas, c='k', s=25, label="Measured gsw")

                    ax.plot_surface(Q, D, gsw_modeled, color="r",edgecolor='none', alpha=0.8, label="BMF Fit")
                    ax.set_xlabel(r"$Q$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=12)
                    ax.set_ylabel(r"$D$ (mmol mol$^{-1}$)", fontsize=12)
                    ax.set_zlabel(r"g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=12)
                    ax.view_init(elev=2, azim=20)

                    ax.set_facecolor('white') 
                    fig.patch.set_facecolor('white') 
                    ax.grid(False,which="major")

                    ax.scatter(Q_meas, D_meas, gsw_meas, c='k', s=25, label="Measured gsw")
                    ax.set_xticks([0,1000,2000])
                    ax.set_zlim([0,max(1,max(gsw_meas)*1.1)])

                    st.pyplot(fig)

                    
                    
                except Exception as e:
                    st.write("Error in fitting:", e)
# ---- PRESSURE-VOLUME MODEL ----
with tabs[2]:

    st.header('Pressure-Volume Curve Fitting')

    uploaded_file = st.file_uploader("Upload your files", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df_pv = pd.read_csv(uploaded_file)
        else:
            df_pv = pd.read_excel(uploaded_file)

        st.success(f"✅ Loaded {len(df_pv)} rows from {uploaded_file.name}")
        st.dataframe(df_pv.head(), hide_index=True)

        data_pv = df_pv

        # Species filtering
        species_col = next((col for col in data_pv.columns if col.lower() == "species"), None)

        if species_col:
            species_list = data_pv[species_col].unique()
            selected_species = st.selectbox("Select a species", species_list, key="pv_species_select")
            data_pv = data_pv[data_pv[species_col] == selected_species]

        # ---- AUTO-DETECT PV COLUMNS ----
        if 'x_colum_pv' not in st.session_state:
            st.session_state.x_colum_pv = None
        if 'y_colum_pv' not in st.session_state:
            st.session_state.y_colum_pv = None

        if st.session_state.x_colum_pv is None and st.session_state.y_colum_pv is None:
            column_candidates = {
                "RWC": [
                    "rwc", "relative_water_content", "relative water content", "Relative Water Content", 
                    "RWC (%)", "RelWC", "RelWC%", "Rel_WC", "Rel_Water", "RWC_percent", "RWC_%", "rwcontent"
                ],
                "Psi": [
                    "psi", "Ψ", "Psi", "water_potential", "water potential", "LeafPsi", "Leaf_Psi", 
                    "leaf_water_potential", "LWP", "WP", "psi_leaf", "Pleaf", "Pressure", "P_leaf","Psi (Bar)","psi(bar)","psi(MPa)","psi (MPa)"
                ]
            }

            def normalize(name):
                return name.lower().replace(" ", "").replace("_", "")

            normalized_cols = {normalize(col): col for col in data_pv.columns}

            for key, candidates in column_candidates.items():
                for c in candidates:
                    norm_c = normalize(c)
                    if norm_c in normalized_cols:
                        if key == "RWC":
                            st.session_state.x_colum_pv = normalized_cols[norm_c]
                        elif key == "Psi":
                            st.session_state.y_colum_pv = normalized_cols[norm_c]
                        break

        # ---- COLUMN SELECTION UI ----
        st.write("### Select Columns for PV Curve Fitting")
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.x_colum_pv = st.selectbox(
                "X (Relative Water Content, RWC)", 
                options=list(data_pv.columns), 
                index=(list(data_pv.columns).index(st.session_state.x_colum_pv)
                       if st.session_state.x_colum_pv in data_pv.columns else 0)
            )

        with col2:
            st.session_state.y_colum_pv = st.selectbox(
                "Y (Water Potential, Ψ)", 
                options=list(data_pv.columns), 
                index=(list(data_pv.columns).index(st.session_state.y_colum_pv)
                       if st.session_state.y_colum_pv in data_pv.columns else 0)
            )

        if st.session_state.x_colum_pv and st.session_state.y_colum_pv:
            x = data_pv[st.session_state.x_colum_pv]
            y = data_pv[st.session_state.y_colum_pv]

            x = pd.to_numeric(x, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')

            valid = x.notna() & y.notna()
            x = x[valid]
            y = y[valid]

            if x.empty or y.empty:
                st.error("Error: selected data columns are empty after filtering. Check column selection or formatting.")
                st.stop()

            st.markdown("---")
            st.write("### Select Water Potential Units")
            unit = st.radio("What unit is the given water potential data in?", ["MPa", "bar", "kPa", "-MPa", "-bar", "-kPa"], horizontal=True)

            if unit == "bar":
                y = 0.1 * y
            elif unit == "-bar":
                y = -0.1 * y
            elif unit == "kPa":
                y = 0.001 * y
            elif unit == "-kPa":
                y = -0.001 * y
            elif unit == "-MPa":
                y = -1.0 * y

            with st.expander("Advanced Options"):
                st.write("Customize additional settings for model fitting.")
                fix_elasticity = st.checkbox("Fix elastic exponent (ε)?")
                
                if fix_elasticity:
                    fixed_elasticity = st.number_input(r"Value", value=1.0, min_value=0.1, max_value=100.0)
                    bounds = ([-5, 0, fixed_elasticity-0.0001], [0, 0.99, fixed_elasticity])
                else:
                    custom_upper_bound = st.checkbox("Set custom upper bound for elastic exponent (ε)?")
                    elasticity_upper_bound = st.number_input(r"Upper bound", value=2.0, min_value=0.5, max_value=100.0) if custom_upper_bound else 2.0
                    bounds = ([-5, 0, 0], [0, 0.99, elasticity_upper_bound])

            if st.button("Fit Model", key="fit_pv"):
                st.latex(r"\psi(R) = -\pi_o \cdot \left(\frac{R - R_{tlp}}{1 - R_{tlp}}\right)^{\epsilon} + \frac{\pi_o}{R}")
                st.subheader("Best Fit Parameters")
                
                p0 = [-1, 0.8, fixed_elasticity-0.0001] if fix_elasticity else [-1, 0.8, 1]

                try:
                    popt, _ = curve_fit(PV, x, y, p0=p0, bounds=bounds)
                    pio, Rtlp, elasticity = popt
                    y_pred = PV(x, *popt)
                    r2 = r_squared(y, y_pred)
                    rmse_val = rmse(y, y_pred)

                    col1, col2, col3 = st.columns(3)
                    with col1: st.latex(r"\pi_o = " + f"{pio:.2f}")
                    with col2: st.latex(r"R_{tlp} = " + f"{Rtlp:.2f}")
                    with col3: st.latex(r"\epsilon = " + f"{elasticity:.2f}")

                    st.latex(r"R^{2} = " + f"{r2:.3f}")
                    st.latex(r"RMSE = " + f"{rmse_val:.3f}")

                    results_df = pd.DataFrame({
                        'species': [selected_species] if species_col else [],
                        'pi_o': [round(pio, 4)],
                        'R_tlp': [round(Rtlp, 4)],
                        'elastic_exponent': [round(elasticity, 4)],
                        'Rsquared': [round(r2, 4)],
                        'RMSE': [round(rmse_val, 4)]
                    })
                    csv = results_df.to_csv(index=False)
                    st.dataframe(results_df, hide_index=True)
                    st.download_button(label="Download Results as CSV", data=csv, file_name="fitted_parameters.csv", mime="text/csv")

                    fig, ax = plt.subplots()
                    ax.plot(x, y, 'ko', label="Measured")
                    xx = np.linspace(0.5, 1, 100)
                    ax.plot(xx, PV(xx, *popt), 'r-', label="Modeled")
                    ax.set_xlabel("Relative Water Content (/)")
                    ax.set_ylabel("Leaf Water Potential (MPa)")
                    if species_col:
                        ax.set_title(selected_species)
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error in fitting: {e}")



# ---- PROSPECT MODEL ----
with tabs[3]:
    st.header("PROSPECT Model")
    st.write("This section will include PROSPECT leaf optical property model fitting.")
