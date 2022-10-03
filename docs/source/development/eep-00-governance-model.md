(eep-00)=


# EEP-00: Governance model & code of conduct

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Janoś Gabler <https://github.com/janosg>`_,                     |
|            | `Hans-Martin von Gaudecker <https://github.com/hmgaudecker>`_,   |
|            | `Annica Gehlen <https://github.com/amageh>`_,                    |
|            | `Sebastian Gsell <https://github.com/segsell>`_,                 |
|            | `Mariam Petrosyan <https://github.com/mpetrosian>`_,             |
|            | `Tobias Raabe <https://github.com/tobiasraabe>`_,                |
|            | `Klara Röhrl <https://github.com/roecla>`_                       |
+------------+------------------------------------------------------------------+
| Status     | Draft                                                            |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2022-04-28                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Purpose

This document formalizes the estimagic code of conduct and governance model. In case
of changes, this document can be updated following the Estimagic Enhancement Proposal
process detailed below.



```{include} ../../../CODE_OF_CONDUCT.md
```

## Estimagic governance model

### Summary

The governance model strives to be lightweight and based on [consensus](https://numpy.org/doc/stable/dev/governance/governance.html#consensus-based-decision-making-by-the-community) of all interested parties. Most work happens in GitHub issues and pull requests (regular
decision process). Any interested party can voice their concerns or veto on proposed
changes. If this happens, the estimagic enhancement proposal (EEP) process can
be used to iterate over proposals until consesus is reached (controversial
decision process). If necessary, members of the steering council can moderate heated
debates and help to broker a consensus.

### Regular decision process

Most changes to estimagic are additions of new functionality or strict improvements
of existing functionality. Such changes can be discussed in GitHub Issues and
Discussions and implemented in pull requests. They do not require an Estimagic
Enhancement Proposal.

Before starting to work on estimagic, contributors should read [how to contribute](how-to)
and the [styleguide](styleguide). They can also reach out to existing contributors if
any help is needed or anything remains unclear. We are all happy to help onboarding new
contributors in any way necessary. For example, we have given introductions to git and
GitHub in the past to help people make a contribution to estimagic.

Pull requests should be opened as soon as work is started. They should contain a good
description of the planned work such that any interested party can participate in the
discussion around the changes. If planned changes turn out to be controversial, their
design should be discussed in an Estimagic Enhancement proposal before the actual
work starts. When the work is finished, the author of a pull
request can request a review. In most cases, previous discussions will show who is a
suitable reviewer. If in doubt, tag [janosg](https://github.com/janosg). Pull requests
can be merged if there is at least one approving review.

Reviewers should be polite, welcoming and helpful to the author of the pull request
who might have spent many hours working on the changes. Major points should be discussed
publicly on GitHub, but very critical feedback or small details can be moved to private
discussions. Video calls can help if a discussion gets stuck. Keep in mind that many
people get notified on every review comment (see [here](https://rgommers.github.io/2019/06/the-cost-of-an-open-source-contribution/) for an execellent discussion).
The code of conduct applies to all interactions related to code reviews.

### Estimagic Enhancement Proposals (EEPs) / Controversial decision process

Large changes to estimagic can be proposed in estimagic enhancement proposals, short
EEPs. They serve the purpose of summarising discussions
that may happen in chats, issues, pull requests, in person, or by any other means.
Simple extensions (like adding new optimizers) do not need to be discussed with such
a formal process.

EEPs are written as markdown documents that become part of the documentation. Opening
an EEP means opening a pull request that adds the markdown document to the documentation.
It is not necessary to already have a working implementations for the planned changes,
even though it might be a good idea to have rough prototypes for solutions to the most
challenging parts.

If the author of an EEP feels that it is ready to be accepted they need to make a
post in our [zulip workspace](https://ose.zulipchat.com) and a comment on the PR that
contains the following information:

1. Summary of all contentious aspects of the EEP and how they have been resolved
2. Every interested party has seven days to comment on the PR proposing the EEP,
   either with approval or objections. While only objections are relevant for the
   decision making process, approvals are a good way to signal interest in the planned
   change and recognize the work of the authors.
3. If there are no unresolved objections after seven days, the EEP will automatically
   be accepted and can be merged.

Note that the Pull Requests that actually implement the proposed enhancements still
require a standard review cycle.

### Steering Council

The Estimagic Steering Council consists of five people who take responsibility for
the future development of estimagic and the estimagic community. Being a member of the
steering council comes with no special rights. The main roles of the steering council
are:

- Facilitate the growth of estimagic and the estimagic community by organizing community
events, identifying funding opportunities and improving the experience
of all community members.
- Develop a roadmap, break down large changes into smaller projects and find contributors
to work on the implementation of these projects.
- Ensure that new contributors are onboarded and assisted and that pull requests are
reviewed in a timely fashion.
- Step in as moderators when discussions get heated, help to achieve consensus on
controversial topics and enforce the code of conduct.
