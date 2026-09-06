import streamlit as st
import auth_db

def _init_session_state():
    """Initialize authentication-related session state keys."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = None
    if "full_name" not in st.session_state:
        st.session_state["full_name"] = None

def is_authenticated():
    """Check if the current user is authenticated."""
    _init_session_state()
    return st.session_state["authenticated"]

def get_current_user():
    """Get the current logged-in user's info."""
    return {
        "username": st.session_state.get("username"),
        "role": st.session_state.get("role"),
        "full_name": st.session_state.get("full_name"),
    }

def logout():
    """Clear session state to log out the user."""
    st.session_state["authenticated"] = False
    st.session_state["username"] = None
    st.session_state["role"] = None
    st.session_state["full_name"] = None

def show_login_form():
    """Render the login form."""
    st.subheader("🔐 Login")
    
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("Login", type="primary", use_container_width=True)
    
    if submit:
        if not username or not password:
            st.error("⚠️ Please fill in all fields.")
            return
        
        success, result = auth_db.authenticate_user(username, password)
        
        if success:
            st.session_state["authenticated"] = True
            st.session_state["username"] = result["username"]
            st.session_state["role"] = result["role"]
            st.session_state["full_name"] = result["full_name"]
            st.success(f"✅ Welcome back, {result['full_name']}!")
            st.rerun()
        else:
            st.error(f"❌ {result}")

def show_registration_form():
    """Render the registration form."""
    st.subheader("📝 Register New Account")
    
    with st.form("register_form", clear_on_submit=True):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="Enter your company email")
        username = st.text_input("Username", placeholder="Choose a username")
        password = st.text_input("Password", type="password", placeholder="Choose a password (min 6 characters)")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        submit = st.form_submit_button("Register", type="primary", use_container_width=True)
    
    if submit:
        # Validation
        if not all([full_name, email, username, password, confirm_password]):
            st.error("⚠️ Please fill in all fields.")
            return
        
        if len(username) < 3:
            st.error("⚠️ Username must be at least 3 characters long.")
            return
        
        if len(password) < 6:
            st.error("⚠️ Password must be at least 6 characters long.")
            return
        
        if password != confirm_password:
            st.error("⚠️ Passwords do not match.")
            return
        
        if "@" not in email or "." not in email:
            st.error("⚠️ Please enter a valid email address.")
            return
        
        # Create user — status set to "approved" so users can login immediately
        # --- APPROVAL SYSTEM (disabled for now): change status="approved" to status="pending" to re-enable ---
        success, message = auth_db.create_user(
            username=username,
            email=email,
            full_name=full_name,
            password=password,
            role="user",
            status="pending"
        )
        
        if success:
            st.success("✅ Account created successfully! You can now login.")
        else:
            st.error(f"❌ {message}")

def show_auth_page():
    """Show the authentication page with Login and Register tabs."""
    _init_session_state()
    
    st.title("🔍 PDF Summarizer")
    st.markdown("---")
    
    # Center the auth form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            "**Welcome!** Please login or register to access the PDF Summarizer tool."
        )
        
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])
        
        with login_tab:
            show_login_form()
        
        with register_tab:
            show_registration_form()
    
    # Footer info
    st.markdown("---")
    st.caption("🔒 Only approved company employees can access this tool. "
               "New registrations require admin approval.")
