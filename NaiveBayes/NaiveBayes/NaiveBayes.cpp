// NaiveBayes.cpp : Defines the entry point for the console application.
//

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>


using PaperID = int64_t;
using Point = std::pair<PaperID, std::string>;
using ClassMetadata = std::pair<std::string, std::vector<Point>>;

std::vector<Point> consume_metadata(const std::string& filename)
{
	std::vector<Point> points;

	std::ifstream metadata{ filename };


	//Read in the "reduced" title with no punctuation, all lowercase, etc


	std::string paper_orig_title;
	std::string paper_title;
	std::string paper_id_raw;
	while (std::getline(metadata, paper_id_raw, '\t'))
	{
		std::getline(metadata, paper_orig_title, '\t');
		std::getline(metadata, paper_title, '\t');

		PaperID paper_id = std::stoll(paper_id_raw, nullptr, 16);

		points.emplace_back(paper_id, paper_title);
		metadata.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	return points;
}

//Extract all words from a vector of metadata and add them to a vocabulary set
void extract_vocabulary(const std::vector<Point>& metadata, std::set<std::string>& vocabulary)
{
	for (const auto& docs : metadata)
	{
		const auto& doc = docs.second;
		std::stringstream tokens{ doc };
		std::string token;
		while (tokens >> token)
		{
			vocabulary.emplace(token);
		}
	}
}

/*
Eigen::VectorXi count_occurrences(const std::vector<Point>& metadata, const std::set<std::string>& vocabulary)
{

}
*/

template <typename T>
using Vec = std::vector<T, Eigen::aligned_allocator<T>>;

Vec<Vec<Eigen::Array2i>> train_all(const std::vector<ClassMetadata>& metadata)
{
	Vec<Vec<Eigen::Array2i>> classes;
	for (const auto& C : metadata)
	{
		Eigen::Array2i indices{ 0, static_cast<int>(C.second.size()) };
		classes.emplace_back(Vec<Eigen::Array2i>{indices});
	}

	return classes;
}

Eigen::Index vocab_to_index(const std::vector<std::string>& vocabulary, const std::string& token)
{
	//Take advantage of sorted vector -- do binary search
	const auto n = vocabulary.size();
	Eigen::Index lowerBound = 0;
	Eigen::Index upperBound = n - 1;

	Eigen::Index midpoint = 0;

	do {
		if (upperBound < lowerBound)
		{
			return -1;
		}
		midpoint = (lowerBound + upperBound) / 2;
		if (vocabulary[midpoint] < token)
		{
			//Take second half
			lowerBound = midpoint + 1;
		}
		else if (token < vocabulary[midpoint])
		{
			//Take first half
			upperBound = midpoint - 1;
		}

	} while (vocabulary[midpoint] != token);

	return midpoint;
}

class MultinomialNaiveBayes
{
	int64_t N = 0;
	const std::vector<std::string> V;
	Eigen::ArrayXd prior;
	Eigen::ArrayXXd condProb;
public:


	MultinomialNaiveBayes(const std::vector<ClassMetadata>& metadata, Vec<Vec<Eigen::Array2i>> trainingIndices, const std::set<std::string>& vocabulary) : V{vocabulary.cbegin(), vocabulary.cend()}
	{
		/*for (size_t i = 0; i < metadata; i++)
		{

		}*/
		const auto n_classes = metadata.size();
		Eigen::ArrayXd Nc{ n_classes };

		for (size_t i = 0; i < n_classes; i++)
		{
			const auto& classIndices = trainingIndices[i];
			int n_docs_in_class = 0;
			for (const auto& indices : classIndices)
			{
				const auto start = indices(0);
				const auto end = indices(1);
				n_docs_in_class += end - start;
			}
			Nc(i) = n_docs_in_class;
			N += n_docs_in_class;
			

		}

		//Now we have N, Nc, and V.  Priors are calculated for all classes here.
		prior = Nc / static_cast<double>(N);

		//Initialize conditional probabilities
		condProb = Eigen::ArrayXXd{ V.size(), n_classes };

		for (size_t i = 0; i < n_classes; i++)
		{
			std::map<std::string, int> occurrences;
			int64_t sumOccurrences = 0;

			for (const auto& point : metadata[i].second)
			{
				const auto& doc = point.second;
				std::stringstream tokens{ doc };
				std::string token;
				while (tokens >> token)
				{

					//TODO: for feature selection, will need to modify sumOccurrences here to only accumulate tokens that are in V
					++occurrences[token];
					++sumOccurrences;
				}
			}

			//Include laplace smoothing
			const auto denominator = sumOccurrences + V.size();

			Eigen::Index tIndex = 0;
			for (const auto& t : V)
			{
				int T_tc;
				try
				{
					T_tc = occurrences.at(t);
				}
				catch (const std::out_of_range&)
				{
					T_tc = 0;
				}

				//Include laplace smoothing
				const auto numerator = T_tc + 1;

				condProb(tIndex, i) = static_cast<double>(numerator) / static_cast<double>(denominator);

				tIndex++;
			}
		}

		//Store probabilities as logs in advance
		prior = prior.log();
		condProb = condProb.log();
	}

	Eigen::Index classify(const std::string& d)
	{
		std::stringstream tokens{ d };

		//TODO: do reductions to lowercase, punctuation, etc, in-program instead, and do it here as well as in the training
		std::string token;
		Eigen::ArrayXd scores = prior;

		//Logs already taken
		while (tokens >> token)
		{
			const auto& res = std::lower_bound(V.cbegin(), V.cend(), token);
			if (*res != token)
			{
				continue;
			}
			const auto t = std::distance(V.cbegin(), res);

			for (Eigen::Index c = 0; c < prior.size(); c++)
			{
				scores(c) += condProb(t, c);
			}
		}
		Eigen::Index winner;
		scores.maxCoeff(&winner);

		return winner;
	}

	//Trains using entire metadata set
	MultinomialNaiveBayes(const std::vector<ClassMetadata>& metadata, const std::set<std::string>& vocabulary) : MultinomialNaiveBayes{ metadata, train_all(metadata), vocabulary } {	}

};

Eigen::Array<Eigen::Array2i, -1, -1> get_m_fold_slices(const std::vector<ClassMetadata>& metadata, int M)
{
	const auto n_classes = metadata.size();
	Eigen::Array<Eigen::Array2i, -1, -1> indices{M, n_classes};

	for (int i = 0; i < n_classes; i++)
	{
		const auto Nc = metadata[i].second.size();
		const auto n_per_fold = Nc / M;
		//const auto n_per_fold_r = Nc % M;

		for (int j = 0; j < M; j++)
		{
			const auto start = j*n_per_fold;


			indices(j, i)(0) = static_cast<int>(start);
			
			//Take care to include remainder in the last fold
			if (j == M - 1)
			{
				const auto end = Nc;
				indices(j, i)(1) = static_cast<int>(end);
			}
			else
			{
				const auto end = (j + 1)*n_per_fold;
				indices(j, i)(1) = static_cast<int>(end);
			}
		}
	}

	return indices;
}

template <typename T>
Vec<Eigen::Array2i> get_m_fold_training_indices(const T& indices, const Eigen::Index exclude)
{
	Vec<Eigen::Array2i> chunks;

	if (exclude != 0)
	{
		Vector2i chunk = { indices(0)(0), indices(exclude - 1)(1) };
		chunks.emplace_back(chunk);
	}

	if (exclude != indices.rows() - 1)
	{
		Vector2i chunk = { indices(exclude + 1)(0), indices(indices.rows() - 1)(1) };
		chunks.emplace_back(chunk);
	}

	return chunks;
}

template <typename T>
Eigen::Array2i get_m_fold_testing_indices(const T& indices, const Eigen::Index include)
{
	return indices.row(include);
}

int __cdecl main(int argc, char* argv[])
{
	std::vector<std::pair<std::string, std::string>> metadata_files;
	int n_features = 0;
	if (argc < 2)
	{
		metadata_files.emplace_back("icse", "metadata/icse_id.txt");
		metadata_files.emplace_back("vldb", "metadata/vldb_id.txt");
	}
	else
	{
		std::string first_arg{ argv[1] };
		int start_index = 1;
		if (first_arg == "--selectfeatures")
		{
			if (argc < 3)
			{
				n_features = 100; //Default to selecting 100 features
				start_index = 2;
			}
			else
			{
				try
				{
					n_features = std::stoi(argv[2]);
					start_index = 3;
					if (n_features < 0)
					{
						n_features = 100;
					}
				}
				catch (...)
				{
					start_index = 2;
				}
			}
		}
		if (first_arg == "--help" || (argc - start_index) % 2 != 0)
		{
			std::cout << "Usage: " << argv[0] << " (--selectfeatures (<n>)) [class1 class1filename class2 class2filename ... classN classNfilename]" << std::endl;
			std::cout << "If no arguments are provided, this command line will be run:\n\t" << argv[0] << " --selectfeatures 100 icse metadata/icse_id.txt vldb metadata/vldb_id.txt" << std::endl;

			return 0;
		}
		for (int i = start_index; i < argc; i += 2)
		{
			const auto* class_name = argv[i];
			const auto* filename = argv[i + 1];
			metadata_files.emplace_back(class_name, filename);
		}

	}

	std::vector<ClassMetadata> metadata;


	//Create vocabulary from all data rather than just the training indices specified
	std::set<std::string> V;
	//extract_vocabulary(metadata[i].second, V);
	for (const auto& file : metadata_files)
	{
		const auto& class_metadata = consume_metadata(file.second);
		metadata.emplace_back(file.first, class_metadata);
		extract_vocabulary(class_metadata, V);
	}

	const auto folds = get_m_fold_slices(metadata, 10);


	MultinomialNaiveBayes all{ metadata , V };
	std::string test = "logic programming environments for large knowledge bases a practical perspective abstract";
	std::cout << test << ": " << all.classify(test);

	//std::vector<std::vector<std::pair<it, it>>> = mFoldCrossValidate(metadata, 10);

    return 0;
}

