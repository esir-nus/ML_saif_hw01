# Business Understanding for P2P Lending Analysis

## 1. Deconstruction of Datasets (From Data Dictionaries)
Using first-principles: Break down to fundamentals—what each dataset represents, its core variables, and how they tie to lending risks.

- **LC (Loans Dataset):** Core loan origination data. Key variables (from LCLP dict screenshot):
  - ListingId: Unique loan ID (primary key for joins).
  - 借款金额 (Loan Amount): Amount borrowed, numeric.
  - 借款期限 (Loan Term): Duration in months.
  - 借款利率 (Interest Rate): Annual rate.
  - 借款成功日期 (Success Date): Date loan funded, timestamp.
  - 初始评级 (Initial Rating): Credit grade (e.g., A-F), categorical.
  - 借款类型 (Loan Type): Category (e.g., ordinary, e-commerce).
  - 是否首标 (First Loan): Yes/No flag.
  - Demographics: 年龄 (Age), 性别 (Gender).
  - Certifications: 手机认证 (Phone), 户口认证 (Household), etc. (success/fail flags).
  - History: 历史成功借款次数 (Successful Loans Count), 历史成功借款金额 (Total Successful Amount), 总待还本金 (Outstanding Principal), 历史正常还款期数 (Normal Repayments), 历史逾期还款期数 (Overdue Repayments).
  - Purpose: Captures borrower profile and loan terms—fundamental for risk assessment (e.g., high overdue history signals risk).

- **LP (Repayments Dataset):** Repayment records per loan period. Key variables (from LCLP dict screenshot):
  - ListingId: Joins to LC.
  - 期数 (Period Number): Installment number.
  - 到期日期 (Due Date): Scheduled repayment date.
  - 还款日期 (Repayment Date): Actual repayment date.
  - 应还本金/利息 (Due Principal/Interest): Amounts due.
  - 已还本金/利息 (Paid Principal/Interest): Amounts paid.
  - Label: Overdue flag (1 if overdue, 0 otherwise)—target for prediction.
  - Purpose: Tracks payment behavior; enables overdue calculation (e.g., if 还款日期 > 到期日期, Label=1).

- **LCIS (Investments Dataset):** Investment details per loan. Key variables (from LCIS dict screenshot):
  - ListingId: Joins to LC.
  - 我的投资金额 (My Investment Amount): Amount invested by user.
  - 当前到期期数 (Current Due Period), 当前还款期数 (Current Repaid Period).
  - 已还/待还本金/利息 (Paid/Outstanding Principal/Interest).
  - 标当前逾期天数 (Current Overdue Days): Days overdue for the loan.
  - 标当前状态 (Current Status): e.g., "已还清" (repaid), "逾期中" (overdue).
  - 上次/下次还款 dates and amounts.
  - recorddate: Record timestamp.
  - Purpose: Adds investor perspective; aggregate for features like total investments or average overdue days per loan.

## 2. Relationships Between Datasets
- **Core Structure:** LC is the hub (one row per loan). LP is many-to-one (multiple repayment periods per ListingId). LCIS is many-to-one (multiple investments per ListingId).
- **Joins:** Merge on ListingId (inner for LP to duplicate loan info per period; left for LCIS aggregates to enrich without loss).
- **Business Logic:** LP provides the target (Label for overdue); LC/LCIS provide features (e.g., rating + investment volume predict risk). Deconstruct: Overdue = f(borrower history, loan terms, investment attraction).

## 3. Key Business Aspects: P2P Lending Risks and Overdue Prediction
- **First-Principles Deconstruction:** P2P lending risk boils down to probability of default/overdue (Label=1). Fundamentals:
  - **Inputs (Features):** Borrower traits (age, certifications, history from LC), loan attrs (amount, rate, term from LC), repayment patterns (due vs. actual from LP), investment health (overdue days, total invested from LCIS—e.g., low investments signal poor loan quality).
  - **Output (Target):** Binary Label from LP (overdue yes/no per period; aggregate for loan-level risk).
  - **Why Predict?** Mitigate losses: High-risk loans (e.g., low rating + high overdue history) can be rejected or priced higher.
- **Risk Insights:** From dicts, certifications (e.g., 征信认证) verify identity, reducing fraud; history metrics quantify past behavior (e.g., high 历史逾期还款期数 = repeat risk). Investments (LCIS) indicate market confidence.
- **Lessons for Modeling:** Imbalance likely (few overdues); focus on interpretable features (e.g., rating impacts most); use for prediction to simulate lending decisions.

This understanding guides the project: Clean/merge to build df, explore for insights, model to predict Label.
