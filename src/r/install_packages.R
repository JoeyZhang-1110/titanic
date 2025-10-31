pkgs <- c("readr","dplyr","stringr","tidyr","caret")
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}
