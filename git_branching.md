Git Branching (https://learngitbranching.js.org/)
---

### Monday, 20 Sep

- commit: improved copy and paste (compressing a commit as a set of changes from one version of the rep to the next, keeping history) A commit has a parent it references changes from.

	useful commands:   

		git commit

- branching in git: pointers to a specific commit. A branch essentially says: "I want to include the work of this commit and all parent commits"
	
	useful commands:  
	
		git branch <name> [<commit>] (assing <commit> and its parents the branch name <name>)
		git branch -f <name> [<commit>] (used for reassigning branch name)
		git checkout ...
		git checkout -b ...

- merging in git: combining work from two different branches together. Creates a commit with two unique parents
 
	useful commands: 
		
		git merge

- git rebase: copying a set of commits, makes the commit log / history of the repository a lot cleaner, two parallel commits can be move in a way that they appear as if they happened sequentially

	useful commands: 

		git rebase ...

- moving around in Git:
	- HEAD: points to the most recent commit reflected in the working tree (mostly points to branch name, detaching the HEAD attaches it to a commit rather than a branch)
	- relative refs:

		moving upwards one commit where `num` specifies the parent (for merged commits): `^<num>`  
		moving upwards a number of times: ~<num\>
		
		useful commands: 
			
			git checkout HEAD^2
			git branch -f main HEAD~3 (reassigning branch to a commit)
			git checkout HEAD~; git checkout HEAD^2; git checkout HEAD~2 
			git switch HEAD~^2~2 (same as row above) 

	- reversing changes: we can use the reset command to locally move a branch backwards (as if the commit had never been made) or the revert command to make a new commit that introduces changes that exaclty reverses the previous commit (used for remote branches)  
	useful commands: 
	
		git reset HEAD~1
		git revert HEAD 

- moving work around:
   * cherry-pick: copy a series of commits below your current location (HEAD)  
		useful commands:

			git cherry-pick <Commit1> <Commit2> ...

   * interactive rebase: essentially rebase but with an interactive shell which lets you select the commits you want to rebase

		useful commands:

			git rebase -i HEAD~4

### Tuesday, 21 Sep
- juggling commits: we want to add small changes to a commit that is way back in our history  
	* 1st possibility: chaning the order of the commits (using rebase) s.t. the commit to be changed is on top. Change the commit (using --amend) and than reorder it to get the correct order again

	  useful commands:

			git rebase -i HEAD~3
			git commit --amend (slight modification)
	
	* 2nd possibility: in order to avoid cumbersome reordering, we can use cherry-pick: we first make a copy of the commit that we want to change, change it, and then copy over the main commit 

		```
        	  C0---C1  main                         C2'''---C3'  main*
                	\                               /
	                 C2  newImage     =>      C0---C1
        	          \                             \
                	  C3  caption*                   C2  newImage
                        	                          \
                                	                   C3  caption

		```
		useful commands:

			git switch main
			git cherry-pick newImage
			git commit --amend
			git cherry-pick caption

- Git tags: possibility of marking something more permanent than a branch (e.g. a major release). They never move as commits are created

	useful commands: 
	
		git tag v1 <hash>

	* Git describe: shows you where you are relative to the closest anchor/tag. 

		useful commands:

			git describe <ref>     =>    <tag>_<numCommits>_g<hash>


#### Git remotes: 
Remotes are just copies of your repository on another computer. It's purpose is:

1. backup
2. making coding social (others can pull in your latest changes and/or contribute)

	* creating remotes: one can use git clone to create local copies of remote repositories. However, in Learn Git Branching, it actually makes a remote repository out of your local one.  

		useful command: 

			git clone 
	
	* git remote branches:  

		* relflecting the state of the remote repositories and help you understand the difference between your local and the published work. 
		* switching to them, you are put into detached `HEAD` mode, meaning you cannot work on these branches directly (the remote branch will only update when the remote updates, e.g. `git switch origin/main; git commit`)
		* naming convention: `<remote name>/<branch name>`. The main remote is mostly called `origin`
	
	* git fetch: fetching data *from* a remote repository. It essentially brings the local representation of the remote rep into sync with the actual remote rep. It
		1. downloads commits which the remote has but are missing on the local rep
		2. update where our remote branches point
		
		3. usually talks to the remote rep through the internet via a protocol like `http://` or `git://`	
		
		4. will not change any of your local files but rather download the download the necessary data to reflect the state of the remote rep
		
		useful command:

			git fetch

	* git pull: incorporating the commits from the remote rep into our own work. This can be done using `git fetch` followed by executing commands like `git cherry-pick origin/main, git rebase origin/main, git merge origin/main` and so on. Git actually provide a command that does this things all at once: `git pull` (corresponds to `git fetch; git merge origin/main`)

		useful commands:

			git pull
			git pull --rebase (corresponds to git fetch; git rebase <branch name>)

	* git push: uploading your changes to a specified remote und updating the remote to incorporate your new commits
		
		* if there is other work published on the remote rep that is not incorporated in your local rep, you first have to pull the latest state in order to push your changerep, you first have to pull the latest state in order to push your changes

		useful commands:

			git push <remote> <place>
			git push <remote> <source>:<destination>

		where `place` specifies where the commits will come from and where the commits will go (branch on the remote `<remote>`; the remote is usually called `origin`). If you choose to push to a destination different to the source, you can use the second command.

### Wednesday, 22 Sep
#### Work flow
1. update main and push work	

		git pull [--rebase]; git push

2. push features to the updated remote
	1. using rebase
	
			git fetch	
			git rebase origin/main feature0
			...
			git rebase feature<num> main
			git push

	2. using merge
		
			git switch main
			git merge feature0
			...
			git push

3. Remote tracking  
	local branches track remote branches. This means that it is an implied merge tartget/push destination (e.g. if you pull `main`, the remote rep will be fetched -> `origin/main` will be updated, and afterwards the local branch `main` will be merged with `origin/main`). Which local branches will be merged with `origin/main` is specified when cloning the rep. However, you can also specify the branch that is tracking with

		git checkout -b totallyNotMain origin/main
		git branch -u origin/main [totallyNotMain] (you can drop totallyNotMain if its currently checked out)

4. Setting up SSH key for gitlab

	https://docs.gitlab.com/ee/ssh/index.html
