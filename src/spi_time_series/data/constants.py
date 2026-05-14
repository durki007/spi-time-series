"""
Constants used in the data module.
"""

# List of valid end activities used to filter out incomplete cases.
VALID_END_ACTIVITIES = [
    "W_Validate application",
    "W_Call after offers",
    "W_Call incomplete files",
    "O_Cancelled",
    "A_Denied",
]

# Number of top variants to keep when filtering infrequent variants in the event log.
TOP_K_VARIANTS = 100
