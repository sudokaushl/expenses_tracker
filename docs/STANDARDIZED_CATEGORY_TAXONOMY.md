# Standardized Category Taxonomy

Derived from the supplied category/subcategory rows. Exact duplicate rows are collapsed; category labels with clear spelling/pluralization errors are normalized. This is a reference taxonomy only—it does not update the database or rewrite transaction labels.

## Canonical pairs

| Category | Subcategories |
|---|---|
| Banking | Account Charges; Debit Card Charges; ECS Charges; Failed Transaction; GST; Interest; MOPS; Refund; Verification Payment |
| Cash Withdrawal / Deposit | Cash Withdrawal; Transfer |
| Charity | Child Care; Devotional; Political |
| Credit Cards | Credit Card Payment |
| Entertainment | Games; Movie; OTT; Refund; Sports |
| Events & Gifts | Anniversary; Birthday; Festival; Gift; Holi; Photoshoot; Wedding |
| Family & Household | Beauty Parlour; Car Wash; Donation; Education Fees; Garbage; Groceries; Lend; Personal; Plot Registration; Pocket Money; Rahul; Repair; Return; Transfer; Wedding |
| Food & Dining | Food Delivery; Juice; Lunch; Office Cafeteria; Office Lunch; Restaurant; Return; Street Food; Tea & Snacks; Vending Machine |
| Fuel | CNG; Gas; Surcharge |
| Healthcare | Checkup; GYN; Medicine; MRI; PDT; Physio; Refund; Tests; Treatment; Ultrasound |
| Housing & Property | Construction; Education Fees; Lend; Personal; Plot Purchase; Plot Registration; Registry; Rent; Return; Scheme |
| Income | Allowance; Salary |
| Investments | Accidental Insurance; Farming; Life Insurance; Mutual Fund; Stocks |
| Lending & Borrowing | Borrow; Lend; Loan |
| Loan & Interest | EMI; Interest; Pre Closure; Pre EMI |
| Misc | Misc; Parking |
| Personal Care | Facial; Hair Cut; Shaving |
| Personal Expenses | Courses; Elearning; Monthly Spending; Photocopy; Sports; Stationery |
| Refunds & Returns | Refund; Return |
| Shopping | Amazon Pay Balance; Antivirus; Apparel; Book; Bottle; Cosmetics; Electronics; Extension Board; Eye Glasses; Footwear; Gaming; Groceries; Guitar; Healthcare; Inverter Trolley; Jewellery; Multivitamins; Phone; Ritual Supplies; Sports; Table Mat; TV Table; Wedding |
| Subscriptions | OTT |
| Taxes | Deduction; Refund |
| Travel | Cab; Metro; Rickshaw; Uber Balance |
| Transfers | Transfer |
| Uncategorized | Driving Session; Phone Sale |
| Utilities & Bills | AWS; Broadband; DTH; Electricity Bill; Fastag; Fuel; Internet; ITR; Mobile |
| Work & Business | Dinner; Farewell; Office - Birthday; Party |
| Work & Office | Office Event |

## Explicit label normalizations

| Source label | Standard label | Reason |
|---|---|---|
| Entetainment | Entertainment | Corrected spelling |
| Refunds & Return | Refunds & Returns | Standardized category pluralization |

## Preserved distinctions / review notes

- Repeated rows are represented once. Subcategories that occur under multiple categories remain under each source category; no automatic reassignment was made. This includes Education Fees, Fuel, Groceries, Interest, Lend, OTT, Personal, Refund, Return, Sports, Transfer, and Wedding.
- Work & Office / Office Event remains distinct from Work & Business because the supplied data does not establish that the categories are interchangeable.
- Family & Household / Rahul is retained verbatim; confirm whether it is an intentional subcategory or a person-specific label.
- Fuel / Fuel remains under Utilities & Bills, separate from Fuel / CNG, Gas, and Surcharge.
- Refund and Return remain separate subcategories under Refunds & Returns.
