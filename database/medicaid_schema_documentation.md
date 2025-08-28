# 🏥 **Medicaid Database Schema Documentation**

## 📋 **Overview**

This document provides comprehensive documentation for the Medicaid Drug Rebate Program (MDRP) database schema. The system manages claims, payments, pricing, and rebate calculations for Medicaid drug programs across different states and quarters.

## 🏗️ **Database Architecture**

The database follows a **claim-centric architecture** where:
- **Claims** represent individual drug rebate requests
- **Claim Lines** contain specific drug and pricing information
- **Payments** track rebate disbursements
- **Programs** define rebate calculation rules and formulas
- **Validation** ensures data quality and business rule compliance

## 📊 **Core Tables**

### **1. Claims Management**

#### **MN_MCD_CLAIM** - Main Claims Table
**Purpose**: Central table for all Medicaid drug rebate claims
**Key Fields**:
- `MCD_CLAIM_ID`: Primary key for claim identification
- `BUS_ELEM_ID`: Business element identifier
- `PROGRAM_ID`: Associated Medicaid program
- `URA_PRICELIST_ID`: URA pricing list reference
- `CLAIM_NUM`: Human-readable claim number
- `REV_NUM`: Revision number for claim updates
- `CLAIM_STATUS`: Current status (Pending, Approved, Rejected, etc.)
- `CLAIM_TYPE`: Type of claim (Original, Adjustment, Conversion)
- `ORIG_QTR`: Original quarter for the claim
- `STATE`: State where claim is filed
- `LABELER`: Drug manufacturer/labeler
- `REBATE_DUE`: Calculated rebate amount due
- `DUE_DATE`: When rebate payment is due
- `ORIG_QTR_END_DATE`: End date of original quarter

**Business Logic**:
- Claims can be adjusted multiple times (tracked by REV_NUM)
- Each claim is associated with a specific Medicaid program and state
- Claims track both calculated and actual rebate amounts
- Claims maintain audit trail with creation/update timestamps

#### **MN_MCD_CLAIM_LINE** - Claim Line Items
**Purpose**: Detailed line items for each claim containing drug-specific information
**Key Fields**:
- `MCD_CLAIM_LINE_ID`: Primary key for claim line
- `CLAIM_ID`: Reference to parent claim
- `PRODUCT_ID`: Drug product identifier
- `URA_PRICELIST_ID`: URA pricing list for this line
- `SYS_CALC_URA`: System-calculated URA amount
- `OVERRIDE_URA`: Manually overridden URA amount
- `INV_URA`: Invoice URA amount
- `INV_UNITS`: Invoice units (quantity)
- `INV_REQ_REBATE`: Invoice required rebate amount
- `REBATE_DUE`: Final rebate amount due for this line
- `URA_CALC_QTR`: Quarter used for URA calculation
- `VALIDATION_STATUS`: Data validation status
- `REPORTED_STATUS`: Reporting status to CMS

**Business Logic**:
- Each claim can have multiple line items (one per drug)
- URA amounts can be system-calculated or manually overridden
- Units and rebate amounts are tracked separately
- Validation ensures data quality and business rule compliance

### **2. Payment Management**

#### **MN_MCD_PAYMENT** - Payment Records
**Purpose**: Tracks rebate payments and disbursements
**Key Fields**:
- `MCD_PAYMENT_ID`: Primary key for payment
- `PAY_NUM`: Payment number
- `STATUS`: Payment status (Pending, Processed, Mailed, etc.)
- `CACHED_REBATE_AMOUNT`: Cached rebate amount
- `CACHED_INTEREST`: Cached interest amount
- `CACHED_TOTAL_AMOUNT`: Total amount including rebate and interest
- `CHECK_NUM`: Check number for payment
- `CHECK_DATE`: Date check was issued
- `PAID_DATE`: Date payment was processed
- `STATE`: State for the payment
- `RECIPIENT_FK`: Recipient information

**Business Logic**:
- Payments aggregate rebates from multiple claims
- Interest is calculated on overdue amounts
- Payment status tracks the entire payment lifecycle
- Cached amounts ensure consistency during processing

#### **MN_MCD_CLAIM_PMT** - Claim Payment Details
**Purpose**: Links claims to payments and tracks applied credits
**Key Fields**:
- `CLAIM_PAYMENT_ID`: Primary key
- `MCD_CLAIM_ID`: Reference to claim
- `MCD_PAYMENT_ID`: Reference to payment
- `APPLIED_CREDIT`: Credit amount applied to this claim
- `PAST_APPLIED_CREDIT`: Previously applied credits
- `CALC_APPLIED_CREDIT`: Calculated credit amount

**Business Logic**:
- Credits can be applied to reduce rebate amounts
- Tracks both past and current credit applications
- Links individual claims to payment records

### **3. Program Management**

#### **MN_MCD_PROGRAM** - Medicaid Programs
**Purpose**: Defines Medicaid drug rebate programs and their rules
**Key Fields**:
- `MCD_PROGRAM_ID`: Primary key for program
- `PROGRAM_SHORT_NAME`: Short name for program identification
- `EXTERNAL_PROGRAM_NAME`: External program name
- `START_CAL_QTR`: Program start quarter
- `END_CAL_QTR`: Program end quarter
- `PROGRAM_TYPE`: Type of program
- `PROGRAM_STATUS`: Current program status
- `DEFAULT_FORMULA_ID`: Default URA calculation formula
- `URA_FORMULA_TEST_FLAG`: Flag for URA formula testing

**Business Logic**:
- Programs define rebate calculation rules
- Each program has specific start/end quarters
- Programs can have different URA calculation formulas
- Programs can be active, inactive, or in testing mode

#### **MN_MCD_PROGRAM_LI** - Program Line Items
**Purpose**: Links programs to specific products and formulas
**Key Fields**:
- `LINE_ITEM_ID`: Primary key
- `PROGRAM_ID`: Reference to program
- `PRODUCT_ID`: Drug product
- `FORMULA_ID`: URA calculation formula
- `START_CAL_QTR`: Start quarter for this line item
- `END_CAL_QTR`: End quarter for this line item

**Business Logic**:
- Programs can have different rules for different products
- Each product can have specific URA calculation formulas
- Line items define the effective date ranges for rules

#### **MN_MCD_PROG_CONFIG** - Program State Configuration
**Purpose**: State-specific program configuration and rules
**Key Fields**:
- `MCD_PROG_STATE_ID`: Primary key
- `MCD_STATE`: State identifier
- `PROGRAM_ID`: Reference to program
- `START_CAL_QTR`: Configuration start quarter
- `END_CAL_QTR`: Configuration end quarter
- `PAYEE_ID`: Payee information
- `MFR_CONTACT_ID`: Manufacturer contact
- `ANALYST_ID`: Assigned analyst
- `INTEREST_FORMULA`: Interest calculation formula
- `OVERDUE_RULE`: Rule for handling overdue amounts

**Business Logic**:
- Each state can have different program configurations
- States can have different interest calculation rules
- States can have different overdue handling rules
- Each state has assigned analysts and contacts

### **4. Pricing and URA Management**

#### **MN_MCD_PRICELIST_PUBLISHED** - URA Pricing Lists
**Purpose**: Published URA pricing lists for different quarters and programs
**Key Fields**:
- `URA_PRICELIST_ID`: Primary key for pricing list
- `URA_PRICELIST_NAME`: Name of the pricing list
- `QUARTER`: Quarter for the pricing list
- `PROGRAM_SHORT_NAME`: Associated program

**Business Logic**:
- Pricing lists are published quarterly
- Each program can have different pricing lists
- Pricing lists contain the URA amounts for drugs

### **5. Validation and Quality Control**

#### **MN_MCD_VALIDATION_MSG** - Validation Messages
**Purpose**: Tracks validation results and business rule compliance
**Key Fields**:
- `MCD_MSG_ID`: Primary key
- `CLAIM_LINE_ID`: Reference to claim line
- `VALIDATION_CLASS_NAME`: Type of validation
- `SEVERITY`: Severity level (Error, Warning, Info)
- `DISPLAY_ORDER`: Order for display
- `RECOM_DISP_UNITS`: Recommended dispute units
- `RECOM_DISPUTE_CODES`: Recommended dispute codes
- `FORMULA_DEF`: Formula definition
- `FORMULA_EXP`: Formula expression
- `INPUT_NAMES`: Input parameter names
- `INPUT_VALUES`: Input parameter values

**Business Logic**:
- Validates business rules and data quality
- Provides recommendations for dispute resolution
- Tracks formula calculations and inputs
- Ensures compliance with Medicaid requirements

#### **MN_MCD_TOLERANCES** - Tolerance Settings
**Purpose**: Defines acceptable tolerance levels for various calculations
**Key Fields**:
- `MCD_TOLERANCES_ID`: Primary key
- `AVG_UNITS_TOL_PERCENT`: Average units tolerance percentage
- `MAX_UNITS_TOL_PERCENT`: Maximum units tolerance percentage
- `REB_RATIO_TOL_PERCENT`: Rebate ratio tolerance percentage
- `REIMB_TOLERANCE_PERCENT`: Reimbursement tolerance percentage

**Business Logic**:
- Defines acceptable variance levels
- Helps identify potential data quality issues
- Supports automated validation processes

### **6. Mass Updates and Corrections**

#### **MN_MCD_MASS_UPDATE** - Mass Update Operations
**Purpose**: Tracks bulk update operations for claims and data corrections
**Key Fields**:
- `MASS_UPDATE_ID`: Primary key
- `MASS_UPDATE_NAME`: Name of the mass update
- `MASS_UPDATE_STATUS`: Current status
- `VALIDATIONS_RUN_FLAG`: Whether validations were run
- `ERROR_COUNT`: Number of errors encountered
- `IMPL_START_DATE`: Implementation start date
- `COMP_DATE`: Completion date

**Business Logic**:
- Tracks bulk operations for data corrections
- Ensures proper validation before implementation
- Maintains audit trail for mass changes

#### **MN_MCD_MU_PRODUCT** - Mass Update Product Details
**Purpose**: Links mass updates to specific products
**Key Fields**:
- `MU_PROD_ID`: Primary key
- `MASS_UPDATE_ID`: Reference to mass update
- `PRODUCT_ID`: Product being updated
- `START_CAL_QTR`: Start quarter for updates
- `END_CAL_QTR`: End quarter for updates
- `ACTION`: Action to be performed
- `FORMULA_ID`: Formula to be applied

**Business Logic**:
- Defines which products are affected by mass updates
- Specifies the effective date ranges for updates
- Tracks the specific actions and formulas applied

### **7. External Credits and Adjustments**

#### **MN_MCD_EXTERNAL_CREDIT** - External Credit Records
**Purpose**: Tracks credits from external sources (CMS, manufacturers, etc.)
**Key Fields**:
- `MCD_EXTR_CREDIT_ID`: Primary key
- `REF_ID`: External reference identifier
- `STATE`: State for the credit
- `PAYEE_ID`: Payee receiving the credit
- `PROG_ID`: Program associated with the credit
- `PERIOD`: Period for the credit
- `LABELER`: Drug manufacturer/labeler
- `NDC`: National Drug Code
- `CREDIT_AMT`: Credit amount
- `CREDIT_STATUS`: Status of the credit

**Business Logic**:
- Tracks credits from external sources
- Credits can reduce rebate amounts due
- Credits are associated with specific programs and periods

### **8. Settlement and Dispute Resolution**

#### **MN_MCD_SETTLEMENT** - Settlement Records
**Purpose**: Tracks settlement agreements and dispute resolutions
**Key Fields**:
- `SETTLEMENT_ID`: Primary key
- `PAID`: Amount paid in settlement
- `DISMISSED_UNITS`: Units dismissed in settlement
- `RESOLVED_UNITS`: Units resolved in settlement
- `ADJUSTMENT_LINE_ID`: Reference to adjustment line
- `SETTLED_LINE_ID`: Reference to settled line

**Business Logic**:
- Tracks settlement agreements
- Records dismissed and resolved units
- Links settlements to specific claim lines

## 🔗 **Key Relationships**

### **Primary Relationships**
1. **Claims → Claim Lines**: One-to-many relationship
2. **Claims → Payments**: Many-to-many through claim payments
3. **Programs → Program Line Items**: One-to-many relationship
4. **Programs → State Configurations**: One-to-many relationship
5. **Claims → Validation Messages**: One-to-many relationship

### **Business Flow**
```
Program Definition → Claim Creation → Validation → Payment → Settlement
       ↓                ↓            ↓          ↓         ↓
   State Config    Claim Lines   Validation   Credits   Disputes
       ↓                ↓            ↓          ↓         ↓
   URA Formulas    URA Calc      Business    External   Resolution
                   Units         Rules       Credits
```

## 🎯 **URA Calculation Process**

### **1. Data Collection**
- Collect invoice data (units, amounts)
- Identify applicable URA pricing
- Determine calculation quarter

### **2. URA Calculation**
- Apply program-specific formulas
- Consider drug type (brand vs. generic)
- Apply CPI adjustments if applicable
- Calculate rebate amounts

### **3. Validation**
- Check against business rules
- Validate unit tolerances
- Ensure formula compliance
- Generate validation messages

### **4. Payment Processing**
- Aggregate rebates from claims
- Apply available credits
- Calculate interest on overdue amounts
- Process payments

## 📊 **Data Quality Considerations**

### **Critical Fields**
- **URA amounts**: Must be accurate for rebate calculations
- **Units**: Must match invoice and payment records
- **Dates**: Must be within valid program periods
- **Status fields**: Must reflect current state accurately

### **Validation Rules**
- **Business rule compliance**: All claims must follow program rules
- **Data consistency**: Related fields must be consistent
- **Tolerance checking**: Amounts must be within acceptable ranges
- **Audit trail**: All changes must be tracked

## 🔍 **Common Query Patterns**

### **1. URA Discrepancy Analysis**
```sql
-- Find claims with URA discrepancies
SELECT 
    cl.MCD_CLAIM_LINE_ID,
    cl.SYS_CALC_URA,
    cl.INV_URA,
    ABS(cl.SYS_CALC_URA - cl.INV_URA) as URA_DIFFERENCE,
    c.CLAIM_STATUS,
    c.ORIG_QTR
FROM MN_MCD_CLAIM_LINE cl
JOIN MN_MCD_CLAIM c ON cl.CLAIM_ID = c.MCD_CLAIM_ID
WHERE ABS(cl.SYS_CALC_URA - cl.INV_URA) > tolerance_threshold
```

### **2. Payment Analysis**
```sql
-- Analyze payment patterns by state and quarter
SELECT 
    c.STATE,
    c.ORIG_QTR,
    SUM(cl.REBATE_DUE) as TOTAL_REBATE_DUE,
    COUNT(*) as CLAIM_COUNT
FROM MN_MCD_CLAIM c
JOIN MN_MCD_CLAIM_LINE cl ON c.MCD_CLAIM_ID = cl.CLAIM_ID
GROUP BY c.STATE, c.ORIG_QTR
ORDER BY c.ORIG_QTR DESC, c.STATE
```

### **3. Validation Issues**
```sql
-- Find validation issues by severity
SELECT 
    vm.SEVERITY,
    vm.VALIDATION_CLASS_NAME,
    COUNT(*) as ISSUE_COUNT
FROM MN_MCD_VALIDATION_MSG vm
JOIN MN_MCD_CLAIM_LINE cl ON vm.CLAIM_LINE_ID = cl.MCD_CLAIM_LINE_ID
WHERE vm.SEVERITY IN ('ERROR', 'WARNING')
GROUP BY vm.SEVERITY, vm.VALIDATION_CLASS_NAME
ORDER BY vm.SEVERITY, ISSUE_COUNT DESC
```

## 🚨 **Business Rules and Constraints**

### **1. Claim Processing Rules**
- Claims must be associated with valid programs
- Claim lines must have valid URA amounts
- Units must be within acceptable tolerance ranges
- All required fields must be populated

### **2. Payment Rules**
- Payments can only be processed for approved claims
- Credits must be applied before payment processing
- Interest calculations must follow state-specific rules
- Payment amounts must match claim totals

### **3. Validation Rules**
- Business rule violations must be resolved before approval
- Formula calculations must be validated
- Data quality checks must pass
- Audit requirements must be met

## 📈 **Performance Considerations**

### **Indexing Strategy**
- Primary keys on all tables
- Foreign key indexes for joins
- Composite indexes on frequently queried combinations
- Date-based indexes for time-series queries

### **Partitioning**
- Consider partitioning by quarter for large tables
- Partition by state for state-specific queries
- Archive old data to improve performance

### **Query Optimization**
- Use appropriate join strategies
- Leverage materialized views for complex calculations
- Implement query result caching where appropriate

---

## 🔗 **Related Documentation**

- **URA Calculation Formulas**: See program configuration tables
- **Business Rules**: See validation message tables
- **State-Specific Rules**: See program state configuration tables
- **Payment Processing**: See payment and claim payment tables

## 📞 **Support and Maintenance**

For questions about this schema or to report issues:
- Review validation messages for data quality issues
- Check business rule compliance in validation tables
- Monitor payment processing through payment status tables
- Use audit trail fields for change tracking

---

*This documentation is maintained by the Medicaid Data Management Team and should be updated when schema changes occur.* 