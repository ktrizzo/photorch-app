import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
tabs = st.tabs(["Pressure-Volume", "Photosynthesis", "Stomatal Conductance", "PROSPECT"])

# ---- PRESSURE-VOLUME MODEL ----
with tabs[0]:

    st.header('Pressure-Volume Curve Fitting')

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data_pv = pd.read_csv(uploaded_file)
        all_data_pv = data_pv
        st.subheader("Uploaded Spreadsheet")
        st.dataframe(data_pv)

        # Species filtering
        species_col = next((col for col in data_pv.columns if col.lower() == "species"), None)

        if species_col:
            species_list = data_pv[species_col].unique()
            selected_species = st.selectbox("Select a species", species_list, key="pv_species_select")
            data_pv = data_pv[data_pv[species_col] == selected_species]


        # Initialize session state for X and Y selection
        if 'x_colum_pv' not in st.session_state:
            st.session_state.x_colum_pv = None
        if 'y_colum_pv' not in st.session_state:
            st.session_state.y_colum_pv = None

        def select_column(col):
            if st.session_state.x_colum_pv is None:
                st.session_state.x_colum_pv = col
            elif st.session_state.y_colum_pv is None and col != st.session_state.x_colum_pv:
                st.session_state.y_colum_pv = col

        def clear_x():
            st.session_state.x_colum_pv = None

        def clear_y():
            st.session_state.y_colum_pv = None

        st.write("### Select X and Y columns")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**X (Relative Water Content, RWC):**")
            if st.session_state.x_colum_pv:
                if st.button(st.session_state.x_colum_pv, key="clear_x_pv", on_click=clear_x):
                    pass
            else:
                st.write("(Click a column below to select X)")
        
        with col2:
            st.write("**Y (Water Potential, Psi):**")
            if st.session_state.y_colum_pv:
                if st.button(st.session_state.y_colum_pv, key="clear_y_pv", on_click=clear_y):
                    pass
            else:
                st.write("(Click a column below to select Y)")
        
        # Show buttons for each column in a horizontal layout
        st.write("### Available Columns")
        button_container = st.container()
        cols = button_container.columns(min(len(data_pv.columns), 5))
        for i, col in enumerate(data_pv.columns):
            with cols[i % len(cols)]:
                if st.button(col, key=f"col_{i}_{col}_pv", on_click=select_column, args=(col,)):
                    pass
        
        # Ensure columns are selected before proceeding
        if st.session_state.x_colum_pv and st.session_state.y_colum_pv:
            x = data_pv[st.session_state.x_colum_pv]
            y = data_pv[st.session_state.y_colum_pv]

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
                    fit_all_species = st.checkbox("Fit each species in file?",key="fit_all_pv")

            if st.button("Fit Model",key="fit_pv"):

                if(fit_all_species):
                    st.latex(r"\psi(R) = -\pi_o \cdot \left(\frac{R - R_{tlp}}{1 - R_{tlp}}\right)^{\epsilon} + \frac{\pi_o}{R}")

                    if fix_elasticity:
                        p0 = [-1, 0.8, fixed_elasticity-0.0001]
                    else:
                        p0 = [-1, 0.8, 1]
                    species_fits = []
                    fig, ax = plt.subplots()

                    for species in species_list:
                        data_pv = all_data_pv
                        species_data_pv = data_pv[data_pv[species_col] == species]
                        x = species_data_pv[st.session_state.x_colum_pv]
                        y = species_data_pv[st.session_state.y_colum_pv]

                        if unit == "bar":
                            y = 0.1 * y   # Convert to MPa
                        
                        if unit == "-bar":
                            y = -0.1 * y  # Convert to MPa

                        if unit == "kPa":
                            y = 0.001 * y   # Convert to MPa

                        if unit == "-kPa":
                            y = -0.001 * y   # Convert to MPa

                        if unit == "-MPa":
                            y = -1.0 * y  # Convert to MPa

                        try:
                            popt, _ = curve_fit(PV, x, y, p0=p0, bounds=bounds)
                            pio, Rtlp, elasticity = popt

                            # Compute R2 and RMSE
                            y_pred = PV(x, *popt)
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
                            ax.plot(xx, PV(xx, *popt), '-', label=f"{species}",color=scatter[0].get_color())

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
                        popt, _ = curve_fit(PV, x, y, p0=p0, bounds=bounds)
                        pio, Rtlp, elasticity = popt  
                        
                        # Compute R2 and RMSE
                        y_pred = PV(x, *popt)
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
                        ax.plot(xx, PV(xx, *popt), 'r-', label="Modeled")
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

    st.header("Stomatal Conductance Modeling Fitting")

    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"],key="sc_file_uploader")
    
    if uploaded_file is not None:
        data_sc = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Spreadsheet")
        st.dataframe(data_sc)

        # Species filtering
        species_col = next((col for col in data_sc.columns if col.lower() == "species"), None)

        if species_col:
            species_list = data_sc[species_col].unique()
            selected_species = st.selectbox("Select a species", species_list, key="sc_species_select")
            data_sc = data_sc[data_sc[species_col] == selected_species]

        st.write("### Select Model")
        

        models = [
            "Buckley, Turnbull, Adams (2012)",
            "Ball, Woodrow, Berry (1987)",
            "Ball-Berry-Leuning (1995)",
            "Medlyn et al. (2011)"
        ]


        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = models[0]

        # Create a placeholder for the selected model text
        selected_model_text = st.empty()

        # Update the placeholder with the latest selection
        selected_model_text.write(f"**Selected Model:** {st.session_state.selected_model}")

        cols = st.columns(len(models))

        for i, model in enumerate(models):
            if cols[i].button(model, key=model, use_container_width=True):
                st.session_state.selected_model = model
                selected_model_text.write(f"**Selected Model:** {st.session_state.selected_model}")  # Update the text immediately


        st.write("### Select Model Input Columns")
        if(st.session_state.selected_model == "Buckley, Turnbull, Adams (2012)"):
            # Initialize session state for X and Y selection
            if 'x_colum_sc' not in st.session_state:
                st.session_state.x_colum_sc = None
            if 'y_colum_sc' not in st.session_state:
                st.session_state.y_colum_sc = None
            if 'z_column' not in st.session_state:
                st.session_state.z_column = None

            def select_column(col):
                if st.session_state.x_colum_sc is None:
                    st.session_state.x_colum_sc = col
                elif st.session_state.y_colum_sc is None and col != st.session_state.x_colum_sc:
                    st.session_state.y_colum_sc = col
                elif st.session_state.z_column is None and col not in [st.session_state.x_colum_sc, st.session_state.y_colum_sc]:
                    st.session_state.z_column = col

            def clear_x():
                st.session_state.x_colum_sc = None

            def clear_y():
                st.session_state.y_colum_sc = None
            
            def clear_z():
                st.session_state.z_column = None

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Q (Leaf Light Flux, PPFD):**")
                if st.session_state.x_colum_sc:
                    if st.button(st.session_state.x_colum_sc, key="clear_x", on_click=clear_x):
                        pass
                else:
                    st.write("(Click a column below to select Q)")
            
            with col2:
                st.write("**D (Leaf to Air Vapor Pressure Difference, VPD):**")
                if st.session_state.y_colum_sc:
                    if st.button(st.session_state.y_colum_sc, key="clear_y", on_click=clear_y):
                        pass
                else:
                    st.write("(Click a column below to select D)")
            
            with col3:
                st.write("**gsw (Stomatal Conductance to Water Vapor):**")
                if st.session_state.z_column:
                    if st.button(st.session_state.z_column, key="clear_z", on_click=clear_z):
                        pass
                else:
                    st.write("(Click a column below to select gsw)")
            
            # Show buttons for each column in a horizontal layout
            st.write("### Available Columns")
            button_container = st.container()
            cols = button_container.columns(min(len(data_sc.columns), 5))
            for i, col in enumerate(data_sc.columns):
                with cols[i % len(cols)]:
                    if st.button(col, key=f"col_{i}_{col}_sc", on_click=select_column, args=(col,)):
                        pass
            
            # Ensure columns are selected before proceeding
            if st.session_state.x_colum_sc and st.session_state.y_colum_sc and st.session_state.z_column:
                x = data_sc[st.session_state.x_colum_sc]
                y = data_sc[st.session_state.y_colum_sc]
                z = data_sc[st.session_state.z_column]

                abs_PAR = 0.85

                # Unit check section
                st.markdown("---")  # Horizontal divider
                st.write("### Select Q Units")
                unit = st.radio("What unit is the given light flux data in?", ["umol/m2/s ambient", "umol/m2/s absorbed", "W/m2 ambient", "W/m2 absorbed"], horizontal=True)   

                if(unit == "umol/m2/s ambient"):
                    x *= abs_PAR
                elif(unit == "W/m2 ambient"):
                    x *= abs_PAR * 4.57 #ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf
                elif(unit == "W/m2 absorbed"):
                    x *= 4.57 

                st.write("### Select D Units")
                unit = st.radio("What unit is the given leaf VPD data in?", ["kPa", "mol/mol", "mmol/mol"], horizontal=True)     

                if(unit == "kPa"):
                    y *= 1000.0 / 101.3
                elif(unit == "mol/mol"):
                    y *= 1000.0

                # Add an Advanced Options expander before the Fit Model button
            with st.expander("Advanced Options"):
                st.write("Customize additional settings for model fitting.")

                # Option to fix elasticity exponent
                set_PAR_abs = st.checkbox("Set leaf PAR absorbtivity (Default = 0.85)")
                
                if set_PAR_abs:
                    abs_PAR = st.number_input(r"Value", value=0.85, min_value=0.1, max_value=1.0)
                    x *= abs_PAR/0.85

                if species_col:
                    fit_all_species = st.checkbox("Fit each species in file?",key="fit_all_sc")    

        if(st.session_state.selected_model == "Ball, Woodrow, Berry (1987)"):
            # Initialize session state for X and Y selection
            if 'x_colum_sc' not in st.session_state:
                st.session_state.x_colum_sc = None
            if 'y_colum_sc' not in st.session_state:
                st.session_state.y_colum_sc = None
            if 'z_column' not in st.session_state:
                st.session_state.z_column = None

            def select_column(col):
                if st.session_state.x_colum_sc is None:
                    st.session_state.x_colum_sc = col
                elif st.session_state.y_colum_sc is None and col != st.session_state.x_colum_sc:
                    st.session_state.y_colum_sc = col
                elif st.session_state.z_column is None and col not in [st.session_state.x_colum_sc, st.session_state.y_colum_sc]:
                    st.session_state.z_column = col

            def clear_x():
                st.session_state.x_colum_sc = None

            def clear_y():
                st.session_state.y_colum_sc = None
            
            def clear_z():
                st.session_state.z_column = None

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**A (Assimilation):**")
                if st.session_state.x_colum_sc:
                    if st.button(st.session_state.x_colum_sc, key="clear_x", on_click=clear_x):
                        pass
                else:
                    st.write("(Click a column below to select A)")
            
            with col2:
                st.write("**Ca (Ambient CO2 Concentration):**")
                if st.session_state.y_colum_sc:
                    if st.button(st.session_state.y_colum_sc, key="clear_y", on_click=clear_y):
                        pass
                else:
                    st.write("(Click a column below to select Ca)")
            
            with col3:
                st.write("**gsw (Stomatal Conductance to Water Vapor):**")
                if st.session_state.z_column:
                    if st.button(st.session_state.z_column, key="clear_z", on_click=clear_z):
                        pass
                else:
                    st.write("(Click a column below to select gsw)")
            
            # Show buttons for each column in a horizontal layout
            st.write("### Available Columns")
            button_container = st.container()
            cols = button_container.columns(min(len(data_sc.columns), 5))
            for i, col in enumerate(data_sc.columns):
                with cols[i % len(cols)]:
                    if st.button(col, key=f"col_{col}", on_click=select_column, args=(col,)):
                        pass
            
            # Ensure columns are selected before proceeding
            if st.session_state.x_colum_sc and st.session_state.y_colum_sc and st.session_state.z_column:
                x = data_sc[st.session_state.x_colum_sc]
                y = data_sc[st.session_state.y_colum_sc]
                z = data_sc[st.session_state.z_column]

            # Unit check section
            st.markdown("---")  # Horizontal divider
            st.write("### Select A Units")
            unit = st.radio("What unit is the given photosynthesis data in?", ["umol/m2/s"], horizontal=True)   
            st.write("### Select Ca Units")
            unit = st.radio("What unit is the given CO2 data in?", ["umol/mol","ppm"], horizontal=True)         

            # Add an Advanced Options expander before the Fit Model button
            with st.expander("Advanced Options"):
                st.write("Customize additional settings for model fitting.")

                # Option to fix elasticity exponent
                set_PAR_abs = st.checkbox("Set leaf PAR absorbtivity")
                
                if set_PAR_abs:
                    abs_PAR = st.number_input(r"Value", value=0.85, min_value=0.1, max_value=1.0)

                if species_col:
                    fit_all_species = st.checkbox("Fit each species in file?")

        if st.button("Fit Model",key = "fit_sc"):
            if(st.session_state.selected_model == "Buckley, Turnbull, Adams (2012)"):
                if species_col:
                    if fit_all_species:
                        st.latex(r"\g_{sw}(Q,D) = \frac{E_m (Q+i_0)}{k+bQ+(Q+i_0)D}")

                        if fix_elasticity:
                            p0 = [-1, 0.8, fixed_elasticity-0.0001]
                        else:
                            p0 = [-1, 0.8, 1]
                        species_fits = []
                        fig, ax = plt.subplots()

                        for species in species_list:
                            data_sc = all_data_sc
                            species_data_sc = data_sc[data_sc[species_col] == species]
                            x = species_data_sc[st.session_state.x_colum_sc]
                            y = species_data_sc[st.session_state.y_colum_sc]

                            try:
                                popt, _ = curve_fit(lambda X, Em, i0, k, b: BTA(X, Em, i0, k, b), 
                                        (x, y), z, p0=p0, bounds=bounds)

                                Em, i0, k, b = popt


                                # Compute R2 and RMSE
                                z_pred = BTA((x,y), *popt)
                                r2 = r_squared(z, z_pred)
                                rmse_val = rmse(z, z_pred)

                                # Store results
                                species_fits.append({
                                    'species': species,
                                    'E_m,': round(Em, 4),
                                    'i_0': round(i0, 4),
                                    'k': round(k, 4),
                                    'b': round(b,4),
                                    'Rsquared': round(r2, 3),
                                    'RMSE': round(rmse_val, 3)
                                })

                                # Plot data and fit
                                scatter = ax.plot(x, y, 'o')
                                xx = np.linspace(0.5, 1, 100)
                                ax.plot(xx, PV(xx, *popt), '-', label=f"{species}",color=scatter[0].get_color())

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
                                'species': [selected_species],
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
                        st.dataframe(results_df)
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

                        
                        Q_meas = x
                        D_meas = y
                        gsw_meas = z

                        Q = np.linspace(0,2000,50)    
                        D = np.linspace(1, 50, 50)   
                        Q,D = np.meshgrid(Q, D)

                        fig = plt.figure(figsize=(10, 10))
                        ax1 = fig.add_subplot(1, 1, 1, projection='3d')

                        gsw_modeled = BTA((Q,D), Em,i0,k,b)


                        ax1.plot_surface(Q, D, gsw_modeled, cmap='YlGn', edgecolor='none', alpha=0.5, label="BMF Fit")
                        ax1.set_xlabel(r"$Q$ ($\mu$mol m$^{-2}$ s$^{-1}$)", fontsize=12)
                        ax1.set_ylabel(r"$D$ (mmol mol$^{-1}$)", fontsize=12)
                        ax1.set_zlabel(r"g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=12)
                        ax1.view_init(elev=15, azim=15)

                        ax1.scatter(Q_meas, D_meas, gsw_meas, c='r', s=25, label="Measured gsw")
                        ax1.set_xticks([0,1000,2000])
                        st.pyplot(fig)


                        # Plot 1:1 reference line
                        gsw_pred = BTA((Q_meas,D_meas), Em,i0,k,b)

                        fig, ax = plt.subplots(figsize=(5, 5))

                        err = np.abs(gsw_meas - gsw_pred) / gsw_meas
                        ax.scatter(gsw_meas, gsw_pred, c="Green", label="Data",s=20)
                        min_val = min(gsw_meas.min(), gsw_pred.min())
                        max_val = max(gsw_meas.max(), gsw_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")

                        ax.set_xlabel(r"Measured g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=13)
                        ax.set_ylabel(r"Modeled g$_{sw}$ (mol m$^{-2}$ s$^{-1}$)", fontsize=13)
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                        
                        
                    except Exception as e:
                        st.write("Error in fitting:", e)

# ---- PROSPECT MODEL ----
with tabs[3]:
    st.header("PROSPECT Model")
    st.write("This section will include PROSPECT leaf optical property model fitting.")
