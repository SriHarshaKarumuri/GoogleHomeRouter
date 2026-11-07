"""Train ticket checker tool implementation."""


def check_train_tickets(origin: str, destination: str, date: str) -> str:
    """Check train ticket availability between origin and destination on a specific date.

    Args:
        origin: Departure station or city name
        destination: Arrival station or city name
        date: Travel date in YYYY-MM-DD format

    Returns:
        Formatted string containing train ticket information
    """
    # Mock data - replace with actual API integration
    mock_results = [
        {
            "train_number": "ICE 123",
            "departure": "08:00",
            "arrival": "12:30",
            "price_eur": 45.50,
            "seats_available": 23,
            "class": "2nd Class",
        },
        {
            "train_number": "IC 456",
            "departure": "10:15",
            "arrival": "14:45",
            "price_eur": 38.00,
            "seats_available": 7,
            "class": "2nd Class",
        },
        {
            "train_number": "RE 789",
            "departure": "14:00",
            "arrival": "18:30",
            "price_eur": 42.00,
            "seats_available": 0,
            "class": "2nd Class",
        },
    ]

    result_lines = [f"Train tickets from {origin} to {destination} on {date}:"]

    if not mock_results:
        return "No trains available for the selected route and date."

    for ticket in mock_results:
        status = f"{ticket['seats_available']} seats available" if ticket["seats_available"] > 0 else "Sold out"
        result_lines.append(
            f"- {ticket['train_number']}: Departs {ticket['departure']}, "
            f"Arrives {ticket['arrival']}, {ticket['class']} - EUR {ticket['price_eur']:.2f} [{status}]"
        )

    return "\n".join(result_lines)
