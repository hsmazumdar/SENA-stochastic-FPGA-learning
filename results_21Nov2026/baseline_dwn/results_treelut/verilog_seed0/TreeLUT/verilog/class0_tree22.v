module class0_tree22(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[324]) & (~i[235])) | ((~i[316]) & i[346] & i[235]) | (i[102] & (~i[346]) & i[235]) | (i[185] & i[324] & (~i[235])) | (i[145] & i[316] & i[346] & i[235]) | ((~i[145]) & i[316] & i[346] & i[235]) | (i[183] & (~i[185]) & i[324] & (~i[235])) | ((~i[183]) & (~i[185]) & i[324] & (~i[235])))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
