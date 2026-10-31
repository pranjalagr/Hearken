-- The food-delivery shop. Fake data only.
-- Money is whole paise (₹450.50 = 45050), so it can never round wrongly.

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL            -- for the identity check (decision 2)
) STRICT;

CREATE TABLE orders (
    -- 4 digits: easy to say out loud, and look-alikes (4512 / 4521) are possible on purpose.
    order_no       INTEGER PRIMARY KEY CHECK (order_no BETWEEN 1000 AND 9999),
    customer_id    INTEGER NOT NULL REFERENCES customers (customer_id),
    amount_paise   INTEGER NOT NULL CHECK (amount_paise > 0),
    payment_method TEXT NOT NULL CHECK (payment_method IN ('upi', 'card', 'cash')),
    address        TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN
                       ('placed', 'preparing', 'out_for_delivery', 'delivered', 'cancelled'))
) STRICT;

CREATE TABLE order_items (
    order_no    INTEGER NOT NULL REFERENCES orders (order_no),
    item        TEXT NOT NULL,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    price_paise INTEGER NOT NULL CHECK (price_paise > 0),   -- price of one unit
    PRIMARY KEY (order_no, item)
) STRICT;

CREATE TABLE refunds (
    refund_id    INTEGER PRIMARY KEY,
    order_no     INTEGER NOT NULL REFERENCES orders (order_no),
    amount_paise INTEGER NOT NULL CHECK (amount_paise > 0),
    -- Cash is paid at the door, so the agent never sends a cash refund.
    method       TEXT NOT NULL CHECK (method IN ('upi', 'card')),
    reason       TEXT NOT NULL,
    token        TEXT,                    -- the token that caused it; empty in plain mode
    at           REAL NOT NULL            -- pretend-clock seconds; negative = before the call
) STRICT;

CREATE TABLE handoffs (
    handoff_id INTEGER PRIMARY KEY,
    order_no   INTEGER REFERENCES orders (order_no),   -- empty if no order was named
    reason     TEXT NOT NULL,
    at         REAL NOT NULL
) STRICT;
