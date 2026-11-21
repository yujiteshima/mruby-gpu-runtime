# CI template for mruby GPU gems

`.github/workflows/ci.yml` in this repository is written so that another mruby GPU gem can copy it. It builds mruby with the gem and runs the gem's tests on GitHub-hosted runners, which have no GPU. Vulkan runs on lavapipe, Mesa's CPU implementation.

## What it runs

| | |
|---|---|
| mruby | 3.4.0 and 4.0.0 |
| Runners | `ubuntu-24.04` (x64) and `ubuntu-24.04-arm` |
| Suites | the gem's own tests, and one suite for each line of `DOWNSTREAM` |

Every suite runs on every mruby version and runner. Each job:

1. installs the toolchain, the Vulkan loader, lavapipe, `glslangValidator` and `vulkaninfo`
2. prints the Vulkan device (`vulkaninfo --summary`), so the log shows that lavapipe was used
3. clones the downstream gems, if the suite has any
4. builds mruby with the `default` gembox, this checkout, the gems in `TEST_GEMS`, and the downstream gems
5. runs the test script and fails unless its output has the line `ALL TESTS PASSED`; the gem's own suite also fails on a `SKIP` line, unless `FAIL_ON_SKIP` is `'false'`

The job summary shows the test counts of each job.

## How to copy it

1. Copy `.github/workflows/ci.yml` to the same path in your gem.
2. Change the values under `Inputs`, near the top of the file:

   | Input | Meaning | Value in this repository |
   |---|---|---|
   | `GEM_NAME` | the name in your `mrbgem.rake` | `mruby-gpu-runtime` |
   | `TEST_FILE` | your test script, relative to the repository root | `test/runtime_test.rb` |
   | `FAIL_ON_SKIP` | whether your own suite fails when the test script prints a `SKIP` line | `'true'` |
   | `TEST_GEMS` | test-only gems inside your repository, listed in every suite | `test/capi` |
   | `DOWNSTREAM` | the gems that build on yours, one suite per line | mruby-gpu-narray and mruby-gpu-kernel |

3. Nothing else needs to change. The matrix of mruby versions and runners is in the `test` job, if you need a different one.

### The test script

The workflow does not use the exit status of the test script. It checks the output: the script must print the line `ALL TESTS PASSED` when every test passes, and must not print it otherwise. `test/runtime_test.rb` in this repository is a small example.

To show the test counts in the job summary, also print a line that starts with `<number> tests, `, for example `14 tests, 14 passed, 0 failed`.

### `FAIL_ON_SKIP`

A test script can leave tests out and still print `ALL TESTS PASSED`. `test/runtime_test.rb` does this for the C API tests in a build without `test/capi`, and prints a line that starts with `SKIP`. With `FAIL_ON_SKIP: 'true'`, the gem's own suite fails when the output has such a line, so the CI cannot pass without those tests.

Set it to `'false'` if your script skips tests on purpose on lavapipe, for example a test that needs a limit lavapipe does not have. The `DOWNSTREAM` suites never check for `SKIP` lines, because their scripts belong to other gems.

### `TEST_GEMS`

Gems that exist only for the tests and live inside your repository: directories relative to the repository root, separated by spaces. Each must have an `mrbgem.rake`. Every suite, the downstream ones too, lists them in the build configuration by path, right after your checkout.

This repository uses `TEST_GEMS: test/capi`. That gem gives `test/runtime_test.rb` a way to call the C API, and building it in every suite shows that it builds next to the downstream gems. A gem with no test-only gems sets `TEST_GEMS: ''`.

### `DOWNSTREAM`

Each line adds one suite. A suite builds other gems on top of your checkout and runs their tests, so a change that breaks them shows up in your pull request, before it is merged.

```
<suite name> <test file> <owner/repository@branch> [<owner/repository@branch> ...]
```

- The suite name appears in the job name.
- The repositories are cloned from GitHub and listed in the build configuration by path, in the order given, after your checkout and `TEST_GEMS`. List a gem after the gems it depends on.
- The test file is relative to the root of the last repository on the line. The test runs from that directory.
- Always name the branch.

This repository uses:

```yaml
DOWNSTREAM: |
  narray test/narray_test.rb yujiteshima/mruby-gpu-narray@main
  kernel test/kernel_test.rb yujiteshima/mruby-gpu-narray@main yujiteshima/mruby-gpu-kernel@main
```

A gem that nothing builds on yet sets `DOWNSTREAM: ''`. Then only its own tests run.

## How dependencies are resolved

- **Your checkout is listed by path, and first.** When a downstream gem names your gem as a GitHub dependency (`spec.add_dependency 'your-gem', github: '...', branch: 'main'`), mruby uses the gem that is already in the build configuration and does not fetch it. So the code under test is the code in the pull request. The build step checks this: it fails if mruby fetched its own copy of `GEM_NAME`.
- **Gems that your gem depends on are not listed.** mruby fetches them the way it does for a user, from the `add_dependency` lines in `mrbgem.rake`. The build step prints the commit of each gem it fetched.
- **Name the branch in a GitHub dependency.** mruby clones `master` when `add_dependency` gives `github:` without `branch:`, and the mruby GPU gems have only `main`.
- **Export your headers if other gems include them.** mruby 4.0 adds a dependency's `include/` only to the gems that list it directly. See `spec.export_include_paths` in this repository's `mrbgem.rake`.

## Notes

- lavapipe's ICD file is `lvp_icd.x86_64.json` on x64 and `lvp_icd.json` on arm64. The workflow finds it with a glob and sets `VK_ICD_FILENAMES`.
- lavapipe checks that the code is correct. It says nothing about speed, or about the limits and behavior of a real GPU driver. Runs on devices (Raspberry Pi 5, Jetson, Apple silicon) are still needed for those.
- The `Read the inputs` job exists because a job matrix cannot read `env`. It turns the inputs into the list of suites.
