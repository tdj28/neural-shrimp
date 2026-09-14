# Zeta: use the existing GitHub work identity

14 September 2026. The human explicitly requested `github-work` rather than
`github-personal` for Git operations on zeta. This amends the authentication
instructions in [the migration record](zeta-20260913.md); Git-over-SSH does not
require a separate GitHub CLI login.

The migrated origins now use the existing work alias, retaining repository
ownership and names:

| Checkout on zeta | Origin, for fetch and push |
|---|---|
| `/data2/neural-shrimp` | `git@github-work:tdj28/neural-shrimp.git` |
| `/data2/butterfly` | `git@github-work:tdj28/butterfly.git` |
| `/data2/agent-skill-documents` | `git@github-work:praxagent/agent-skill-documents.git` |

The existing `github-work` alias resolves to `github.com`, user `git`, with
`IdentityFile ~/.ssh/id_ed25519_github_work` and `IdentitiesOnly yes`.
No SSH key, passphrase, account token or global SSH configuration was changed
or copied. There were no explicit origin push URLs to override these settings.

## Actual authentication status

The work-key files exist with restricted private-key permissions. GitHub
accepts the public key, but the encrypted private key cannot sign in the
current noninteractive shell. No `SSH_AUTH_SOCK` is exported there. The
existing desktop agent refuses signing, and the existing OpenSSH agent at
`/run/user/1000/openssh_agent` currently has no loaded identities.

The neural-shrimp repository now has this narrowly scoped local Git setting:

```text
core.sshCommand = ssh -o IdentityAgent=/run/user/1000/openssh_agent
```

The coordinating butterfly task applied the same setting to its repositories.
This selects the existing agent for Git; the `github-work` alias still selects
the work identity. No new agent daemon was created. These local Git settings
are not tracked repository files and must be checked in any separately created
clone or managed worktree after handoff.

From a terminal on the Mac, unlock the work key on zeta:

```sh
ssh -t zeta 'SSH_AUTH_SOCK=/run/user/1000/openssh_agent ssh-add ~/.ssh/id_ed25519_github_work'
```

Enter the passphrase only at the terminal prompt, not in a chat. This needs
repeating if the agent restarts or the identity is removed or expires. After
unlock, verify the authenticated GitHub account, private-repository reads,
and a push dry-run of an existing unchanged branch. Public-key recognition
alone does not establish the account identity or repository write permission.
Those checks are still pending as this record is prepared. Do not claim that
the failed initial access checks succeeded, and do not reset or overwrite a
branch to test permissions.

## Conversation handoff still pending

The research files, environment and saved checks are ready under `/data2`.
The desktop app still lists only local projects. Connect `zeta` under
**Settings > Connections > SSH**, save `/data2/butterfly` and
`/data2/neural-shrimp` as remote projects, then hand off each conversation
through its run-location control after its current response finishes.
This conversation is attached to `butterfly`, while its research is at
`/data2/neural-shrimp`. Verify hostname, actual working directory and access
to those absolute paths after handoff. The worktree-root caveat in the
earlier migration record still applies.

The Codex computer-use surface is blocked and the current chat cannot call
its own handoff tool, so the final in-app steps remain human actions.
Scientific results, frozen attempts and numerical thresholds are unchanged;
the retained strict Linux replay failure remains a separate research issue.
