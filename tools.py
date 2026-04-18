from datetime import datetime
from langchain_core.tools import tool

@tool
def calculate_project_deadline(current_date: str = None) -> str:
    """
    Calculates the time remaining until the Capstone Project deadline.
    The deadline is strictly April 21, 11:59 PM.
    """
    # Use 2026 as the year since the current year context is 2026
    deadline = datetime(2026, 4, 21, 23, 59, 59)
    now = datetime.now()
    
    if now > deadline:
        return "The deadline (April 21, 11:59 PM) has already passed!"
        
    time_remaining = deadline - now
    days = time_remaining.days
    hours, remainder = divmod(time_remaining.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    
    return f"The Capstone Project deadline is April 21, 11:59 PM. You have {days} days, {hours} hours, and {minutes} minutes remaining."
