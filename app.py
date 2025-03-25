import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the helper functions
def smax(a, b, k):
    h = np.maximum(0, 1 - np.abs(a - b) / (4 * k))
    return np.maximum(a, b) + h**2 * k

def smin(a, b, k):
    h = np.maximum(0, 1 - np.abs(a - b) / (4 * k))
    return np.minimum(a, b) - h**2 * k

# Define the mechanistic model
def mechPV(x, pio, Rtlp, elasticity):
    R = x
    pi = pio / R
    P = - pio * smax(0, (R - Rtlp) / (1 - Rtlp), 0.01)**elasticity
    psi = P + pi
    return psi

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
tabs = st.tabs(["Pressure-Volume", "Photosynthesis", "Stomatal Conductance", "PROSPECT"])

# ---- PRESSURE-VOLUME MODEL ----
with tabs[0]:
    st.header('Pressure-Volume Curve Fitting')

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        all_data = data
        st.subheader("Uploaded Spreadsheet")
        st.dataframe(data)

        # Species filtering
        species_col = next((col for col in data.columns if col.lower() == "species"), None)

        if species_col:
            species_list = data[species_col].unique()
            selected_species = st.selectbox("Select a species", species_list)
            data = data[data[species_col] == selected_species]


        # Initialize session state for X and Y selection
        if 'x_column' not in st.session_state:
            st.session_state.x_column = None
        if 'y_column' not in st.session_state:
            st.session_state.y_column = None

        def select_column(col):
            if st.session_state.x_column is None:
                st.session_state.x_column = col
            elif st.session_state.y_column is None and col != st.session_state.x_column:
                st.session_state.y_column = col

        def clear_x():
            st.session_state.x_column = None

        def clear_y():
            st.session_state.y_column = None

        st.write("### Select X and Y columns")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**X (Relative Water Content, RWC):**")
            if st.session_state.x_column:
                if st.button(st.session_state.x_column, key="clear_x", on_click=clear_x):
                    pass
            else:
                st.write("(Click a column below to select X)")
        
        with col2:
            st.write("**Y (Water Potential, Psi):**")
            if st.session_state.y_column:
                if st.button(st.session_state.y_column, key="clear_y", on_click=clear_y):
                    pass
            else:
                st.write("(Click a column below to select Y)")
        
        # Show buttons for each column in a horizontal layout
        st.write("### Available Columns")
        button_container = st.container()
        cols = button_container.columns(min(len(data.columns), 5))
        for i, col in enumerate(data.columns):
            with cols[i % len(cols)]:
                if st.button(col, key=f"col_{col}", on_click=select_column, args=(col,)):
                    pass
        
        # Ensure columns are selected before proceeding
        if st.session_state.x_column and st.session_state.y_column:
            x = data[st.session_state.x_column]
            y = data[st.session_state.y_column]

            # Unit check section
            st.markdown("---")  # Horizontal divider
            st.write("### Select Water Potential Units")
            unit = st.radio("What unit is the given water potential data in?", ["MPa", "bar", "kPa", "-MPa", "-bar","-kPa"], horizontal=True)

            if unit == "bar":
                y = 0.1 * y   # Convert to MPa
            
            if unit == "-bar":
                y = -0.1 * y  # Convert to MPa

            if unit == "kPa":
                y = 0.001 * y   # Convert to MPa

            if unit == "kPa":
                y = -0.001 * y   # Convert to MPa

            if unit == "-MPa":
                y = -1.0 * y  # Convert to MPa

            # Add an Advanced Options expander before the Fit Model button
            with st.expander("Advanced Options"):
                st.write("Customize additional settings for model fitting.")

                # Option to fix elasticity exponent
                fix_elasticity = st.checkbox("Fix elastic exponent (ε)?")
                
                if fix_elasticity:
                    fixed_elasticity = st.number_input(r"Value", value=1.0, min_value=0.1, max_value=100.0)
                    bounds = ([-5, 0, fixed_elasticity-0.0001], [0, 0.99, fixed_elasticity])  # Fix ε in bounds
                else:
                    # Option to set custom upper bound for elasticity
                    custom_upper_bound = st.checkbox("Set custom upper bound for elastic exponent (ε)?")
                    if custom_upper_bound:
                        elasticity_upper_bound = st.number_input(r"Upper bound", value=2.0, min_value=0.5, max_value=100.0)
                    else:
                        elasticity_upper_bound = 2.0  # Default upper bound

                    bounds = ([-5, 0, 0], [0, 0.99, elasticity_upper_bound])  # Apply upper bound
                
                if species_col:
                    fit_all_species = st.checkbox("Fit each species in file?")

            if st.button("Fit Model"):

                if(fit_all_species):
                    st.latex(r"\psi(R) = -\pi_o \cdot \left(\frac{R - R_{tlp}}{1 - R_{tlp}}\right)^{\epsilon} + \frac{\pi_o}{R}")

                    if fix_elasticity:
                        p0 = [-1, 0.8, fixed_elasticity-0.0001]
                    else:
                        p0 = [-1, 0.8, 1]
                    species_fits = []
                    fig, ax = plt.subplots()

                    for species in species_list:
                        data = all_data
                        species_data = data[data[species_col] == species]
                        x = species_data[st.session_state.x_column]
                        y = species_data[st.session_state.y_column]

                        if unit == "bar":
                            y = 0.1 * y   # Convert to MPa
                        
                        if unit == "-bar":
                            y = -0.1 * y  # Convert to MPa

                        if unit == "kPa":
                            y = 0.001 * y   # Convert to MPa

                        if unit == "kPa":
                            y = -0.001 * y   # Convert to MPa

                        if unit == "-MPa":
                            y = -1.0 * y  # Convert to MPa

                        try:
                            popt, _ = curve_fit(mechPV, x, y, p0=p0, bounds=bounds)
                            pio, Rtlp, elasticity = popt

                            # Compute R² and RMSE
                            y_pred = mechPV(x, *popt)
                            r2 = r_squared(y, y_pred)
                            rmse_val = rmse(y, y_pred)

                            # Store results
                            species_fits.append({
                                'species': species,
                                'pi_o': round(pio, 4),
                                'R_tlp': round(Rtlp, 4),
                                'elastic_exponent': round(elasticity, 4),
                                'Rsquared': round(r2, 3),
                                'RMSE': round(rmse_val, 3)
                            })

                            # Plot data and fit
                            scatter = ax.plot(x, y, 'o')
                            xx = np.linspace(0.5, 1, 100)
                            ax.plot(xx, mechPV(xx, *popt), '-', label=f"{species}",color=scatter[0].get_color())

                        except Exception as e:
                            st.write(f"Error fitting {species}: {e}")

                    # Display Table
                    results_df = pd.DataFrame(species_fits)
                    st.subheader("Best Fit Parameters for All Species")
                    st.dataframe(results_df)

                    # Download Button
                    csv = results_df.to_csv(index=False)
                    st.download_button("Download Results as CSV", data=csv, file_name="fitted_parameters.csv", mime="text/csv")

                    # Finalize Plot
                    ax.set_xlabel("Relative Water Content (/)")
                    ax.set_ylabel("Leaf Water Potential (MPa)")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.latex(r"\psi(R) = -\pi_o \cdot \left(\frac{R - R_{tlp}}{1 - R_{tlp}}\right)^{\epsilon} + \frac{\pi_o}{R}")
                    st.subheader("Best Fit Parameters")
                    
                    if fix_elasticity:
                        p0 = [-1, 0.8, fixed_elasticity-0.0001]
                    else:
                        p0 = [-1, 0.8, 1]

                    try:
                        popt, _ = curve_fit(mechPV, x, y, p0=p0, bounds=bounds)
                        pio, Rtlp, elasticity = popt  
                        
                        # Compute R² and RMSE
                        y_pred = mechPV(x, *popt)
                        r2 = r_squared(y, y_pred)
                        rmse_val = rmse(y, y_pred)

                        # Display fitted parameters and metrics
                        col1, col2, col3 = st.columns(3)
                        with col1: st.latex(r"\pi_o = " + f"{pio:.2f}")  
                        with col2: st.latex(r"R_{tlp} = " + f"{Rtlp:.2f}")
                        with col3: st.latex(r"\epsilon = " + f"{elasticity:.2f}")
                        
                        st.latex(r"R^{2} = " + f"{r2:.3f}")
                        st.latex(r"RMSE = " + f"{rmse_val:.3f}")

                        # Prepare downloadable results
                        if(species_col):
                            results_df = pd.DataFrame({
                                'species': [selected_species],
                                'pi_o': [round(pio,4)],
                                'R_tlp': [round(Rtlp,4)],
                                'elastic_exponent': [round(elasticity,4)],
                                'Rsquared': [round(r2,4)],
                                'RMSE': [round(rmse_val,4)]
                            })
                        else:
                            results_df = pd.DataFrame({
                                'pi_o': [round(pio,4)],
                                'R_tlp': [round(Rtlp,4)],
                                'elastic_exponent': [round(elasticity,4)],
                                'Rsquared': [round(r2,4)],
                                'RMSE': [round(rmse_val,4)]
                            })
                        csv = results_df.to_csv(index=False)
                        st.dataframe(results_df)
                        st.download_button(label="Download Results as CSV", data=csv, file_name="fitted_parameters.csv", mime="text/csv")
                        

                        # Plot results
                        fig, ax = plt.subplots()
                        ax.plot(x, y, 'ko', label="Measured")
                        xx = np.linspace(0.5, 1, 100)
                        ax.plot(xx, mechPV(xx, *popt), 'r-', label="Modeled")
                        ax.set_xlabel("Relative Water Content (/)")
                        ax.set_ylabel("Leaf Water Potential (MPa)")
                        if species_col:
                            ax.set_title(selected_species)
                        ax.legend()
                        st.pyplot(fig)
                        
                        
                    except Exception as e:
                        st.write("Error in fitting:", e)

# ---- PHOTOSYNTHESIS MODEL ----
with tabs[1]:
    st.header("Photosynthesis Model")
    st.write("This section will include photosynthesis model fitting.")

# ---- STOMATAL CONDUCTANCE ----
with tabs[2]:
    st.header("Stomatal Conductance")
    st.write("This section will include stomatal conductance model fitting.")

# ---- PROSPECT MODEL ----
with tabs[3]:
    st.header("PROSPECT Model")
    st.write("This section will include PROSPECT leaf optical property model fitting.")
