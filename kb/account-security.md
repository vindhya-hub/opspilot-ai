# Account Security and Access Recovery

## Suspicious login alerts

A suspicious login alert is treated as a possible account compromise and handled as a priority security case. The account is secured by signing out every active session, forcing a password reset and reviewing the audit log for unrecognised activity. If the audit log shows data access or changes from an unknown location, the Security team takes ownership and contacts the account owner directly.

## Lost two-factor device

Two-factor authentication can be recovered with the recovery codes saved during setup. Without recovery codes, two-factor login is only reset after identity verification by an account admin or the billing contact, and never on the basis of an email request alone. Verified recovery requests are completed within one business day.

## SSO login problems

SSO redirect loops are usually caused by a mismatched ACS URL or entity ID in the identity provider configuration, or by clock skew between servers. Admins can still sign in with their password through the SSO bypass link while the configuration is corrected. The SSO settings page lists the exact values the identity provider must use.

## Admin access and ownership changes

Admin access for a former employee is removed after a request from another verified admin on the same account. Account ownership is transferred with the ownership transfer form, which the current billing contact confirms.

## Password reset emails

Password reset emails are sent from no-reply@example.com and expire after one hour. A reset email that never arrives is usually blocked by a spam filter or a suppressed address, which support can clear on request.
