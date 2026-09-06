import streamlit as st
import auth_db
import pandas as pd

def show_admin_panel():
    """Render the admin panel in the sidebar. Only visible to admin users."""
    st.header("👑 Admin Panel")
    
    # --- Pending Approvals ---
    pending_users = auth_db.get_pending_users()
    
    if pending_users:
        st.subheader(f"⏳ Pending Approvals ({len(pending_users)})")
        
        for user in pending_users:
            with st.container(border=True):
                st.markdown(f"**{user['full_name']}** (`{user['username']}`)")
                st.caption(f"📧 {user['email']}")
                st.caption(f"📅 Registered: {user['created_at']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve", key=f"approve_{user['username']}", use_container_width=True):
                        auth_db.update_user_status(user['username'], 'approved')
                        st.success(f"Approved {user['username']}")
                        st.rerun()
                with col2:
                    if st.button("❌ Reject", key=f"reject_{user['username']}", use_container_width=True):
                        auth_db.update_user_status(user['username'], 'rejected')
                        st.warning(f"Rejected {user['username']}")
                        st.rerun()
    else:
        st.info("✅ No pending approvals")
    
    st.divider()
    
    # --- All Users ---
    st.subheader("👥 All Users")
    
    all_users = auth_db.get_all_users()
    
    if all_users:
        # Stats
        approved_count = sum(1 for u in all_users if u['status'] == 'approved')
        pending_count = sum(1 for u in all_users if u['status'] == 'pending')
        rejected_count = sum(1 for u in all_users if u['status'] == 'rejected')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Approved", approved_count)
        col2.metric("Pending", pending_count)
        col3.metric("Rejected", rejected_count)
        
        # User list
        for user in all_users:
            status_emoji = {"approved": "🟢", "pending": "🟡", "rejected": "🔴"}.get(user['status'], "⚪")
            role_emoji = "👑" if user['role'] == 'admin' else "👤"
            
            with st.expander(f"{status_emoji} {role_emoji} {user['full_name']} (`{user['username']}`)"):
                st.markdown(f"**Email:** {user['email']}")
                st.markdown(f"**Role:** {user['role']}")
                st.markdown(f"**Status:** {user['status']}")
                st.markdown(f"**Registered:** {user['created_at']}")
                
                # Don't allow admin to delete themselves
                current_user = st.session_state.get("username")
                if user['username'] != current_user:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Toggle role
                        new_role = "admin" if user['role'] == "user" else "user"
                        if st.button(
                            f"Make {new_role.title()}", 
                            key=f"role_{user['username']}", 
                            use_container_width=True
                        ):
                            auth_db.update_user_role(user['username'], new_role)
                            st.success(f"Changed {user['username']} to {new_role}")
                            st.rerun()
                    
                    with col2:
                        if st.button(
                            "🗑️ Delete", 
                            key=f"delete_{user['username']}", 
                            use_container_width=True,
                            type="secondary"
                        ):
                            auth_db.delete_user(user['username'])
                            st.success(f"Deleted {user['username']}")
                            st.rerun()
                else:
                    st.caption("ℹ️ This is your account")
    else:
        st.info("No users found.")
