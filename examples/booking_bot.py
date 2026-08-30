"""Restaurant booking bot — shows multi-tool agent with session."""

import asyncio

from agentino import Agent, Session, tool

# Simulated database
_bookings: list[dict] = []
_menu = [
    {"name": "Margherita", "price": 12, "cuisine": "italian"},
    {"name": "Salmon Teriyaki", "price": 18, "cuisine": "japanese"},
    {"name": "Caesar Salad", "price": 10, "cuisine": "american"},
    {"name": "Pad Thai", "price": 14, "cuisine": "thai"},
]


@tool
def search_menu(query: str, cuisine: str = "any") -> str:
    """Search restaurant menu items by name or cuisine."""
    results = [
        item
        for item in _menu
        if query.lower() in item["name"].lower()
        or (cuisine != "any" and cuisine.lower() == item["cuisine"])
    ]
    if not results:
        return "No items found."
    return "\n".join(f"- {r['name']} (${r['price']}, {r['cuisine']})" for r in results)


@tool
def check_availability(date: str, party_size: int) -> str:
    """Check if tables are available for a given date and party size."""
    # Simulated: always available for parties <= 6
    if party_size > 6:
        return f"Sorry, no tables available for {party_size} on {date}. Max party size is 6."
    return f"Tables available for {party_size} on {date} at: 6:00 PM, 7:30 PM, 9:00 PM"


@tool(timeout=10)
def create_booking(date: str, time: str, party_size: int, name: str, phone: str) -> str:
    """Create a restaurant reservation."""
    booking = {
        "id": len(_bookings) + 1,
        "date": date,
        "time": time,
        "party_size": party_size,
        "name": name,
        "phone": phone,
    }
    _bookings.append(booking)
    return (
        f"Booking confirmed! ID: #{booking['id']} — {name}, {party_size} guests, {date} at {time}"
    )


maria = Agent(
    instructions="""You are Maria, a warm and professional restaurant booking assistant.

You help guests:
- Browse the menu (use search_menu)
- Check table availability (use check_availability)
- Make reservations (use create_booking — always ask for name and phone first)

Be friendly, concise, and proactive. If someone wants to book, check availability first,
then confirm the details before creating the booking.""",
    tools=[search_menu, check_availability, create_booking],
)


async def chat() -> None:
    """A conversation loop. `Agent.run` is a coroutine, so every turn is
    awaited — calling it without awaiting prints the coroutine object and
    never reaches the model."""
    session = Session("./sessions/demo-guest.jsonl")

    print("Maria: Hi! I'm Maria. How can I help you today?\n")
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            reply = await maria.run(user_input, session=session)
            print(f"\nMaria: {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nMaria: Goodbye! Hope to see you soon.")
            break


if __name__ == "__main__":
    asyncio.run(chat())
