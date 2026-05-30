"""
Constants used in the data module.
"""

# A list of all event names in our event log
EVENT_NAMES = [
    "W_Validate application",
    "W_Call after offers",
    "W_Call incomplete files",
    "W_Complete application",
    "W_Handle leads",
    "O_Created",
    "O_Create Offer",
    "O_Sent (mail and online)",
    "A_Validating",
    "A_Concept",
    "A_Create Application",
    "A_Accepted",
    "A_Complete",
    "O_Returned",
    "A_Incomplete",
    "O_Cancelled",
    "A_Submitted",
    "A_Pending",
    "O_Accepted",
    "A_Cancelled",
    "O_Refused",
    "A_Denied",
    "W_Assess potential fraud",
    "O_Sent (online only)",
    "W_Shortened completion",
    "W_Personal Loan collection",
]


# List of valid end activities used to filter out incomplete cases.
VALID_END_ACTIVITIES = [
    "W_Validate application",
    "W_Call after offers",
    "W_Call incomplete files",
    "O_Cancelled",
    "A_Denied",
]

# These events define the three possible outcomes for traces. We know no trace contains more than one of these.
OUTCOME_EVENTS = [
    "A_Pending",
    "A_Denied",
    "A_Cancelled",
]

# Number of top variants to keep when filtering infrequent variants in the event log.
TOP_K_VARIANTS = 100
