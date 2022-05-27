class SimpleFeatureExtractor:

    def get_features(self, parser_state):
        """
        Take in all the parser state information and return features.
        Your features should be autograd.Variable objects of embeddings.

        :param parser_state the ParserState object for the current parse (giving access
            to the stack and input buffer)
        :return A list of autograd.Variable objects, which are the embeddings of your
            features
        """
        # STUDENT
        # hint: parser_state.stack_peek_n
        stac = parser_state.stack_peek_n(1)[0][2]
        inps = parser_state.input_buffer_peek_n(2)
        inp1 = inps[0][2]
        inp2 = inps[1][2]
        return [stac,inp1,inp2]
        raise NotImplementedError

        # END STUDENT
