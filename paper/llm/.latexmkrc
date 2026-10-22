# Refresh only from source-save times, not the compilation clock.
my $revision_status = system('python3', '../../tools/update_llm_revision.py', '--root', '.');
die "Unable to update manuscript revision date\n" if $revision_status != 0;
